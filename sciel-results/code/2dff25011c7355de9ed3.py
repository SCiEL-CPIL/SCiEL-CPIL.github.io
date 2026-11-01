"""Football pool covering code K_3(7,1). Bitset-based greedy + local search."""
import time
import random
import numpy as np
from itertools import product

_DEADLINE = 150.0
_MARGIN = 8.0


def construct(q, n, radius):
    start = time.time()
    
    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start, start + _DEADLINE - _MARGIN)
    
    # Fallback: tetracode product construction
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]
    
    return [list(w) for w in product(range(q), repeat=n)]


def _solve_ternary_7_1(start_time, deadline):
    q, n, radius = 3, 7, 1
    N = q ** n  # 2187
    
    # Precompute digit extraction
    power = [1] * n
    for i in range(1, n):
        power[i] = power[i - 1] * q
    
    def digits_of(x):
        d = [0] * n
        for i in range(n):
            d[i] = x % q
            x //= q
        return d
    
    # Precompute ball (neighbors) for each word as a list of word indices
    neighbors = []
    for i in range(N):
        d = digits_of(i)
        b = [i]
        for pos in range(n):
            cur = d[pos]
            for v in range(q):
                if v != cur:
                    b.append(i + (v - cur) * power[pos])
        neighbors.append(b)
    
    # Precompute ball sizes (should all be 1 + n*(q-1) = 1 + 7*2 = 15)
    ball_size = len(neighbors[0])  # 15
    
    # Precompute coverers: coverers[pt] = list of codeword indices that cover pt
    coverers = [[] for _ in range(N)]
    for cw_idx in range(N):
        for pt in neighbors[cw_idx]:
            coverers[pt].append(cw_idx)
    
    # Precompute ball bitsets (Python int with N bits)
    ball_bitsets = [0] * N
    for i in range(N):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        ball_bitsets[i] = mask
    
    all_mask = (1 << N) - 1
    
    best_codewords = None
    best_size = N + 1  # impossible
    
    def _greedy_criticality(seed_val):
        """Greedy with criticality-based ordering."""
        rng = random.Random(seed_val)
        
        # Coverage deficit: initially all words need 1 cover
        # We track which words are covered
        covered_mask = 0
        codewords = []
        is_chosen = [False] * N
        
        # For criticality: weight of a word = number of codewords that could cover it
        # (i.e., len(coverers[word])). Words with fewer coverers are more critical.
        # Actually all words have the same number of coverers = ball_size = 15.
        # So criticality must be dynamic: as we add codewords, the "effective"
        # number of remaining choices decreases.
        
        # Alternative: use a priority based on how many uncovered words a candidate covers,
        # but weighted by the scarcity of their coverers.
        
        while covered_mask != all_mask:
            if time.time() > deadline - 5:
                break
            
            best_i = -1
            best_score = -1
            
            # Score: number of newly covered words, with bonus for covering words
            # that have few remaining coverers
            for i in range(N):
                if is_chosen[i]:
                    continue
                new_bits = ball_bitsets[i] & ~covered_mask
                if new_bits == 0:
                    continue
                cnt = new_bits.bit_count()
                if cnt == 0:
                    continue
                # Criticality bonus: for each newly covered word, add 1/len(coverers[word])
                # But this is expensive. Use a simpler heuristic:
                # Just use count, with random tie-breaking
                score = cnt
                if score > best_score or (score == best_score and best_i == -1):
                    best_score = score
                    best_i = i
                elif score == best_score and best_i != -1:
                    # Random tie-break
                    if rng.random() < 0.5:
                        best_i = i
            
            if best_i == -1:
                break
            
            codewords.append(best_i)
            is_chosen[best_i] = True
            covered_mask |= ball_bitsets[best_i]
        
        # Ensure full coverage
        if covered_mask != all_mask:
            uncovered = all_mask & ~covered_mask
            while uncovered:
                # Pick random uncovered word, add a coverer
                # Find an uncovered word
                w = (uncovered & -uncovered).bit_length() - 1
                # Pick a random coverer of w
                candidates = [c for c in coverers[w] if not is_chosen[c]]
                if not candidates:
                    candidates = coverers[w]
                c = rng.choice(candidates)
                codewords.append(c)
                is_chosen[c] = True
                covered_mask |= ball_bitsets[c]
                uncovered = all_mask & ~covered_mask
                if time.time() > deadline - 2:
                    break
        
        return codewords
    
    def _local_search(codewords, cover_count, deadline_local):
        """Local search: removals, 1-for-1 swaps, 2-for-1 exchanges."""
        rng = random.Random(int(time.time() * 1000) % 100000)
        improved = True
        no_improve = 0
        max_no_improve = 30
        
        while improved and time.time() < deadline_local:
            improved = False
            no_improve += 1
            
            # Pass 1: Remove redundant codewords
            i = 0
            while i < len(codewords):
                if time.time() > deadline_local:
                    return codewords, cover_count
                cw = codewords[i]
                b = neighbors[cw]
                # Check if all points in ball have count > 1
                min_cnt = min(cover_count[pt] for pt in b)
                if min_cnt > 1:
                    for pt in b:
                        cover_count[pt] -= 1
                    codewords.pop(i)
                    improved = True
                    no_improve = 0
                else:
                    i += 1
            
            if no_improve > max_no_improve:
                break
            
            # Pass 2: 1-for-1 swaps
            if time.time() > deadline_local:
                return codewords, cover_count
            
            for idx in range(len(codewords)):
                if time.time() > deadline_local:
                    return codewords, cover_count
                
                old_cw = codewords[idx]
                b_old = neighbors[old_cw]
                
                # Find critical points: points in ball(old_cw) with count == 1
                critical = [pt for pt in b_old if cover_count[pt] == 1]
                
                if not critical:
                    continue
                
                # Find candidates that cover all critical points
                # Use intersection of coverers
                candidate_set = set(coverers[critical[0]])
                for pt in critical[1:]:
                    candidate_set.intersection_update(coverers[pt])
                
                # Remove old_cw and existing codewords
                candidate_set.discard(old_cw)
                for cw in codewords:
                    candidate_set.discard(cw)
                
                if not candidate_set:
                    continue
                
                # Pick best candidate: minimize increase in redundancy
                candidates_list = list(candidate_set)
                if len(candidates_list) > 30:
                    rng.shuffle(candidates_list)
                    candidates_list = candidates_list[:30]
                
                best_new = -1
                best_cost = float('inf')
                
                for new_cw in candidates_list:
                    b_new = neighbors[new_cw]
                    # Cost: sum of cover_count over b_new (lower is better)
                    cost = sum(cover_count[pt] for pt in b_new)
                    if cost < best_cost:
                        best_cost = cost
                        best_new = new_cw
                
                if best_new == -1:
                    continue
                
                # Perform swap
                codewords[idx] = best_new
                for pt in b_old:
                    cover_count[pt] -= 1
                for pt in neighbors[best_new]:
                    cover_count[pt] += 1
                improved = True
                no_improve = 0
                
                # After swap, try to remove redundants
                j = 0
                while j < len(codewords):
                    cw2 = codewords[j]
                    b2 = neighbors[cw2]
                    if min(cover_count[pt] for pt in b2) > 1:
                        for pt in b2:
                            cover_count[pt] -= 1
                        codewords.pop(j)
                    else:
                        j += 1
            
            # Pass 3: 2-for-1 exchanges (remove 2, add 1)
            if time.time() > deadline_local:
                return codewords, cover_count
            
            # Try: for each pair of codewords, see if we can replace them with one
            if len(codewords) > 2:
                # Sample a few pairs to keep it fast
                num_pairs = min(20, len(codewords) * (len(codewords) - 1) // 2)
                for _ in range(num_pairs):
                    if time.time() > deadline_local:
                        break
                    i1, i2 = rng.sample(range(len(codewords)), 2)
                    cw1, cw2 = codewords[i1], codewords[i2]
                    b1, b2 = neighbors[cw1], neighbors[cw2]
                    
                    # Points uniquely covered by cw1 or cw2 (count==1 and in respective ball)
                    # After removing both, points that would be uncovered:
                    # pts where count==1 and (pt in b1 or pt in b2)
                    # But we need to be careful: a point in both b1 and b2 with count==2
                    # would become count==0 after removal.
                    
                    # Find all points that would be uncovered after removing both
                    uncovered_after = set()
                    for pt in b1:
                        if cover_count[pt] == 1:
                            uncovered_after.add(pt)
                    for pt in b2:
                        if cover_count[pt] <= 2:  # Could be 1 (only in b2) or 2 (in both)
                            if cover_count[pt] == 1:
                                uncovered_after.add(pt)
                            elif cover_count[pt] == 2 and pt in b1:
                                # Count was 2, both cw1 and cw2 cover it, so after removal count=0
                                uncovered_after.add(pt)
                    
                    if not uncovered_after:
                        # Can remove both without adding
                        for pt in b1:
                            cover_count[pt] -= 1
                        for pt in b2:
                            cover_count[pt] -= 1
                        # Remove in reverse order to keep indices valid
                        for rem_idx in sorted([i1, i2], reverse=True):
                            codewords.pop(rem_idx)
                        improved = True
                        no_improve = 0
                        break
                    
                    # Find a single codeword that covers all uncovered_after
                    if len(uncovered_after) <= 5:  # Limit search
                        uncovered_list = list(uncovered_after)
                        candidate_set = set(coverers[uncovered_list[0]])
                        for pt in uncovered_list[1:]:
                            candidate_set.intersection_update(coverers[pt])
                        candidate_set.discard(cw1)
                        candidate_set.discard(cw2)
                        for cw in codewords:
                            candidate_set.discard(cw)
                        
                        if candidate_set:
                            # Pick best
                            best_new = -1
                            best_cost = float('inf')
                            for new_cw in list(candidate_set)[:20]:
                                b_new = neighbors[new_cw]
                                cost = sum(cover_count[pt] for pt in b_new)
                                if cost < best_cost:
                                    best_cost = cost
                                    best_new = new_cw
                            
                            if best_new != -1:
                                # Perform 2-for-1
                                for pt in b1:
                                    cover_count[pt] -= 1
                                for pt in b2:
                                    cover_count[pt] -= 1
                                for pt in neighbors[best_new]:
                                    cover_count[pt] += 1
                                # Remove cw1 and cw2
                                for rem_idx in sorted([i1, i2], reverse=True):
                                    codewords.pop(rem_idx)
                                codewords.append(best_new)
                                improved = True
                                no_improve = 0
                                break
    
        return codewords, cover_count
    
    # Multi-restart
    num_restarts = 5
    per_restart_time = (deadline - start_time) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        seed_val = 42 + r * 1000
        deadline_local = min(deadline - 2, start_time + (r + 1) * per_restart_time)
        
        # Greedy initialization
        codewords = _greedy_criticality(seed_val)
        
        # Build coverage count
        cover_count = np.zeros(N, dtype=np.int16)
        for cw in codewords:
            for pt in neighbors[cw]:
                cover_count[pt] += 1
        
        # Local search
        codewords, cover_count = _local_search(codewords, cover_count, deadline_local)
        
        # Update best
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = list(codewords)
        
        # Early exit if we hit the known best
        if best_size <= 186:
            break
    
    # Final fallback: ensure we have a valid solution
    if best_codewords is None or len(best_codewords) == 0:
        # Use tetracode product as fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]
    
    # Convert to digit lists
    seen = set()
    final = []
    for c in best_codewords:
        if c in seen:
            continue
        seen.add(c)
        final.append(digits_of(c))
    
    # Verify validity (quick check)
    if len(final) > 0:
        cov = np.zeros(N, dtype=np.int8)
        for c in best_codewords:
            for pt in neighbors[c]:
                cov[pt] += 1
        if cov.min() < 1:
            # Invalid, use fallback
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]
    
    return final

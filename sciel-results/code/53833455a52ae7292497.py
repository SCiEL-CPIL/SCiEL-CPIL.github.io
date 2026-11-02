import time
import itertools
import random
import numpy as np
from collections import defaultdict

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius `radius`.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)
    else:
        return _generic_construct(q, n, radius, deadline)

def _construct_3_7_1(deadline):
    """
    Specialized solver for q=3, n=7, radius=1.
    Uses Greedy Set Cover + Iterated Local Search with Swaps.
    """
    q, n, radius = 3, 7, 1
    N = q ** n  # 2187
    
    # 1. Precomputation
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute ball members (list of point indices) for each word
    ball_members = []
    for i in range(N):
        w = all_words[i]
        members = [i]
        for pos in range(n):
            orig_val = w[pos]
            for val in range(q):
                if val != orig_val:
                    new_word = list(w)
                    new_word[pos] = val
                    members.append(word_to_idx[tuple(new_word)])
        ball_members.append(members)
        
    # Precompute bitsets for balls (for greedy init)
    ball_bitsets = []
    for members in ball_members:
        mask = 0
        for m in members:
            mask |= (1 << m)
        ball_bitsets.append(mask)
        
    # Precompute coverers: coverers[point] = list of word indices that cover point
    coverers = [[] for _ in range(N)]
    for w_idx, members in enumerate(ball_members):
        for p in members:
            coverers[p].append(w_idx)
            
    # 2. Greedy Initialization
    # We want to find a set of words covering all N points.
    uncovered_mask = (1 << N) - 1
    chosen_indices = []
    is_chosen = [False] * N
    
    # Optimization: Sort words by some heuristic? Or just standard greedy.
    # Standard greedy: pick word covering most uncovered points.
    # To speed up, we can maintain a count of uncovered points covered by each word.
    # But N=2187 is small. Scanning all N words each step is O(N^2) ~ 4.7M ops.
    # With ~180 steps, that's ~800M ops. Might be too slow in Python.
    
    # Better Greedy:
    # Use a priority queue or just random sampling?
    # Let's try a "Lazy" greedy:
    # Maintain `uncovered_set` as a Python set for fast membership, but bitset is faster for intersection.
    # Let's stick to bitsets but optimize the scan.
    
    # Actually, for N=2187, a simple scan might take ~1-2 seconds per step? No.
    # `bit_count()` is fast.
    # Let's try to be smart:
    # We can pre-compute a list of words.
    # At each step, find i maximizing `bit_count(ball_bitsets[i] & uncovered_mask)`.
    
    # To avoid O(N) scan every time, we can use a "best-first" approach with a heap?
    # But the "best" changes as `uncovered_mask` changes.
    
    # Let's just do the scan, but break early if we find a perfect cover? Unlikely.
    # We'll add a time check.
    
    steps = 0
    while uncovered_mask != 0:
        if time.time() > deadline - 60:
            break
            
        best_i = -1
        best_count = 0
        
        # Optimization: Only consider words not already chosen?
        # Yes, adding a duplicate is useless.
        
        for i in range(N):
            if is_chosen[i]:
                continue
            # Calculate newly covered points
            new_covered = ball_bitsets[i] & uncovered_mask
            if new_covered:
                cnt = new_covered.bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_i = i
                    # Early exit if we cover everything?
                    if cnt == uncovered_mask.bit_count():
                        break
        
        if best_i == -1:
            # Should not happen if code is valid, but if we ran out of time in greedy
            # and didn't finish, we need to finish it.
            # Fallback: Randomly pick any word that covers something?
            # Or just break and fix later?
            # Let's just break and rely on the "ensure validity" step later.
            break
        
        chosen_indices.append(best_i)
        is_chosen[best_i] = True
        uncovered_mask &= ~ball_bitsets[best_i]
        steps += 1
        
    # Ensure Validity: If greedy didn't finish (due to time), add random words until covered.
    if uncovered_mask != 0:
        random.seed(42)
        while uncovered_mask != 0:
            i = random.randint(0, N - 1)
            if not is_chosen[i]:
                chosen_indices.append(i)
                is_chosen[i] = True
            uncovered_mask &= ~ball_bitsets[i]
            if time.time() > deadline - 10:
                break

    # 3. Local Search (Iterated Local Search with Swaps)
    
    # Initialize coverage counts
    cover_count = [0] * N
    for idx in chosen_indices:
        for p in ball_members[idx]:
            cover_count[p] += 1
            
    best_size = len(chosen_indices)
    best_solution = list(chosen_indices)
    
    # Helper: Try to remove redundant codewords
    def try_reduce(cw_list, counts):
        removed = 0
        i = 0
        while i < len(cw_list):
            cw = cw_list[i]
            # Check if all points in ball(cw) have count > 1
            redundant = True
            for p in ball_members[cw]:
                if counts[p] <= 1:
                    redundant = False
                    break
            if redundant:
                for p in ball_members[cw]:
                    counts[p] -= 1
                cw_list.pop(i)
                removed += 1
            else:
                i += 1
        return removed

    # Initial reduction
    try_reduce(chosen_indices, cover_count)
    if len(chosen_indices) < best_size:
        best_size = len(chosen_indices)
        best_solution = list(chosen_indices)
        
    # SA / ILS Loop
    random.seed(123)
    no_improvement_count = 0
    max_no_improvement = 1000 # Reset perturbation if stuck
    
    while time.time() < deadline - 2:
        # 1. Try Reduction
        if try_reduce(chosen_indices, cover_count) > 0:
            no_improvement_count = 0
            if len(chosen_indices) < best_size:
                best_size = len(chosen_indices)
                best_solution = list(chosen_indices)
            continue # Restart loop after modification
        
        # 2. Try Swaps
        # Pick a random codeword to try swapping
        if not chosen_indices:
            break
            
        idx = random.randint(0, len(chosen_indices) - 1)
        old_cw = chosen_indices[idx]
        
        # Find critical points: points in ball(old_cw) with count == 1
        critical_points = []
        for p in ball_members[old_cw]:
            if cover_count[p] == 1:
                critical_points.append(p)
                
        if not critical_points:
            # Should have been removed by reduction
            # Force remove if time is tight? No, just skip.
            continue
            
        # Find candidates that cover ALL critical points
        # Intersection of coverers[p] for p in critical_points
        # Start with coverers of first critical point
        if len(critical_points) == 1:
            candidates = set(coverers[critical_points[0]])
        else:
            candidates = set(coverers[critical_points[0]])
            for p in critical_points[1:]:
                candidates.intersection_update(coverers[p])
        
        # Remove old_cw and existing codewords from candidates
        candidates.discard(old_cw)
        for cw in chosen_indices:
            candidates.discard(cw)
            
        if not candidates:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Perturbation: Randomly swap a few words to escape local min
                # This is a "kick"
                for _ in range(5):
                    if not chosen_indices: break
                    i = random.randint(0, len(chosen_indices) - 1)
                    old = chosen_indices[i]
                    # Find any candidate for old
                    cp = [p for p in ball_members[old] if cover_count[p] == 1]
                    if cp:
                        cand = set(coverers[cp[0]])
                        for p in cp[1:]:
                            cand.intersection_update(coverers[p])
                        cand.discard(old)
                        for c in chosen_indices:
                            cand.discard(c)
                        if cand:
                            new = random.choice(list(cand))
                            # Perform swap
                            for p in ball_members[old]:
                                cover_count[p] -= 1
                            for p in ball_members[new]:
                                cover_count[p] += 1
                            chosen_indices[i] = new
                            # Check validity? It might be invalid now if we didn't cover all CP.
                            # But we picked from intersection of CP coverers, so it's valid.
                no_improvement_count = 0
            continue
        
        # Pick a candidate
        # Heuristic: Pick one that covers the most "extra" points (points not in CP)?
        # Or just random to explore.
        cand_list = list(candidates)
        random.shuffle(cand_list)
        
        # Try up to 5 candidates
        swap_accepted = False
        for new_cw in cand_list[:5]:
            # Simulate swap
            # Decrement old
            for p in ball_members[old_cw]:
                cover_count[p] -= 1
            # Increment new
            for p in ball_members[new_cw]:
                cover_count[p] += 1
                
            # Check if this swap allows a reduction
            # We don't want to call try_reduce immediately as it modifies the list.
            # Instead, check if ANY codeword is now redundant.
            
            # Optimization: Only check codewords that might be affected.
            # The swap affects points in B(old) U B(new).
            # Redundancy of a codeword x depends on counts in B(x).
            # So we only need to check x if B(x) intersects B(old) U B(new).
            # This is complex. Let's just run try_reduce on a COPY? No, too slow.
            
            # Alternative: Just perform the swap in the main list, then try_reduce.
            # If try_reduce removes something, we have an improvement.
            # If not, we revert the swap.
            
            # Swap in list
            chosen_indices[idx] = new_cw
            
            # Try to reduce
            if try_reduce(chosen_indices, cover_count) > 0:
                # Improvement!
                swap_accepted = True
                no_improvement_count = 0
                if len(chosen_indices) < best_size:
                    best_size = len(chosen_indices)
                    best_solution = list(chosen_indices)
                break
            else:
                # No improvement. Revert swap.
                # Revert list
                chosen_indices[idx] = old_cw
                # Revert counts
                for p in ball_members[new_cw]:
                    cover_count[p] -= 1
                for p in ball_members[old_cw]:
                    cover_count[p] += 1
                # Note: try_reduce didn't change anything, so counts are consistent with old_cw.
                # Wait, try_reduce might have removed something? No, it returned 0.
                # So counts are back to pre-swap state?
                # Let's verify:
                # Pre-swap: Counts C.
                # Swap: C - B(old) + B(new).
                # try_reduce: No removal. Counts remain C - B(old) + B(new).
                # Revert: Counts - B(new) + B(old) = C. Correct.
        
        if not swap_accepted:
            no_improvement_count += 1
            
    # Final Result
    # Ensure we return the best solution found
    # Convert indices to words
    result = [list(all_words[idx]) for idx in best_solution]
    
    # Ensure distinctness
    seen = set()
    final_result = []
    for w in result:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            final_result.append(w)
            
    return final_result

def _generic_construct(q, n, radius, deadline):
    """
    Generic greedy + reduction for arbitrary q, n, radius.
    """
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls
    ball_members = []
    for i in range(total_words):
        w = all_words[i]
        members = set()
        # Enumerate all words within Hamming distance `radius`
        for dist in range(radius + 1):
            for positions in itertools.combinations(range(n), dist):
                for deltas in itertools.product(range(1, q), repeat=dist):
                    new_word = list(w)
                    for pos, d in zip(positions, deltas):
                        new_word[pos] = (w[pos] + d) % q
                    members.add(word_to_idx[tuple(new_word)])
        ball_members.append(list(members))
    
    # Greedy set cover
    uncovered_set = set(range(total_words))
    chosen = []
    
    while uncovered_set and time.time() < deadline - 30:
        best_ball = -1
        best_count = -1
        for i in range(total_words):
            count = sum(1 for m in ball_members[i] if m in uncovered_set)
            if count > best_count:
                best_count = count
                best_ball = i
        if best_ball == -1 or best_count <= 0:
            break
        chosen.append(best_ball)
        for m in ball_members[best_ball]:
            if m in uncovered_set:
                uncovered_set.discard(m)
        if not uncovered_set:
            break
    
    # Reduction
    coverage = [[] for _ in range(total_words)]
    for ci, ball_idx in enumerate(chosen):
        for m in ball_members[ball_idx]:
            coverage[m].append(ci)
    
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for ci in range(len(chosen) - 1, -1, -1):
            redundant = True
            for m in ball_members[chosen[ci]]:
                if len(coverage[m]) == 1 and coverage[m][0] == ci:
                    redundant = False
                    break
            if redundant:
                for m in ball_members[chosen[ci]]:
                    coverage[m].remove(ci)
                chosen.pop(ci)
                changed = True
    
    result = [list(all_words[idx]) for idx in chosen]
    seen = set()
    final_result = []
    for w in result:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            final_result.append(w)
    return final_result

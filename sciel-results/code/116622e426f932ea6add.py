import time
import random
import sys
import itertools
import numpy as np

def construct(q, n, radius):
    # Handle the specific task: K_3(6,1)
    # The evaluator passes q, n, radius.
    # If not the specific case, fall back to a simple greedy or direct sum.
    if not (q == 3 and n == 6 and radius == 1):
        # Fallback: Direct sum of tetracode if q=3, n>=4
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic greedy for other cases (slow, but safe)
            all_words = list(itertools.product(range(q), repeat=n))
            code = []
            covered = set()
            for w in all_words:
                if w not in covered:
                    code.append(list(w))
                    for i in range(n):
                        for v in range(q):
                            if v == w[i]:
                                continue
                            new_w = list(w)
                            new_w[i] = v
                            covered.add(tuple(new_w))
                    covered.add(w)
            return code

    # Specific case: q=3, n=6, radius=1
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit

    total = q ** n
    powers = [q ** p for p in range(n)]

    # Precompute neighbors: neighbors[i] is a list of indices j such that dist(i,j) <= 1
    # There are 1 + n*(q-1) = 1 + 6*2 = 13 neighbors per word.
    neighbors = [[] for _ in range(total)]
    for i in range(total):
        digits = [(i // powers[p]) % q for p in range(n)]
        neighbors[i].append(i)
        for p in range(n):
            for alt in range(q):
                if alt != digits[p]:
                    new_i = i + (alt - digits[p]) * powers[p]
                    neighbors[i].append(new_i)

    # Convert to sets for fast membership testing in swap logic
    neighbor_sets = [set(nei) for nei in neighbors]

    # Phase 1: Greedy Set Cover to get an initial code
    uncovered = np.ones(total, dtype=np.int8)
    code_indices = set()
    counts = np.zeros(total, dtype=np.int32)

    # Precompute neighbor lists for faster access in greedy
    # Greedy: pick codeword that covers most uncovered words
    while uncovered.sum() > 0:
        if time.time() > deadline:
            break

        best_candidate = -1
        best_score = 0

        # Optimization: Only check candidates that might cover uncovered words?
        # For 729 words, full scan is fast (~13 * 729 = ~9k ops per iter).
        for c in range(total):
            score = 0
            for u in neighbors[c]:
                if uncovered[u]:
                    score += 1
            if score > best_score:
                best_score = score
                best_candidate = c

        if best_score == 0:
            break

        code_indices.add(best_candidate)
        for u in neighbors[best_candidate]:
            counts[u] += 1
            uncovered[u] = 0

    # Check if we have a valid cover
    if uncovered.sum() != 0:
        # Fallback to tetracode construction if greedy failed (shouldn't happen)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    best_size = len(code_indices)
    best_code = list(code_indices)

    def can_remove(c_idx):
        for nb in neighbors[c_idx]:
            if counts[nb] <= 1:
                return False
        return True

    def get_min_count(c_idx):
        return min(counts[nb] for nb in neighbors[c_idx])

    # Local Search Loop
    # We try to reduce the size by removing redundant codewords or swapping to enable removals.
    
    # To speed up removal checks, we can maintain a list of "removable" candidates.
    # But simple iteration is fine for N=729.
    
    T = 1.0
    cooling_rate = 0.999

    while time.time() < deadline:
        removed_any = False
        
        # 1. Try to remove any redundant codeword
        # Iterate in random order to break symmetries
        code_list = list(code_indices)
        random.shuffle(code_list)
        
        for c in code_list:
            if can_remove(c):
                for nb in neighbors[c]:
                    counts[nb] -= 1
                code_indices.remove(c)
                removed_any = True
                if len(code_indices) < best_size:
                    best_size = len(code_indices)
                    best_code = list(code_indices)
                break

        if removed_any:
            continue

        # 2. If no removal possible, try swaps
        # A swap (c_out, c_in) is valid if after swapping, all neighbors of c_out 
        # that are not neighbors of c_in are still covered by >= 2 codewords (so c_out removal doesn't break coverage? No, c_out is removed, so its unique neighbors must be covered by others).
        # Actually, if we swap, we remove c_out and add c_in.
        # Validity: For all u in neighbors[c_out] \ neighbors[c_in], counts[u] must be >= 2 (so that removing c_out doesn't make counts[u] 0).
        
        found_swap = False
        code_list = list(code_indices)
        random.shuffle(code_list)
        
        for c_out in code_list:
            if time.time() > deadline:
                break
                
            # Identify critical neighbors of c_out (those with count 1)
            # If c_out has any neighbor with count 1, that neighbor MUST be covered by c_in.
            critical_nbs = [u for u in neighbors[c_out] if counts[u] == 1]
            
            # We need c_in to cover all critical_nbs.
            # Also, c_in should ideally cover many other low-count neighbors to improve removability.
            
            # Score c_in:
            # 1. Must cover all critical_nbs.
            # 2. Prefer c_in that covers many neighbors of c_out (overlap) to keep counts high?
            #    Actually, we want to increase the minimum count of the resulting set.
            #    Let's just find valid swaps first.
            
            valid_candidates = []
            for c_in in range(total):
                if c_in in code_indices:
                    continue
                if c_in == c_out:
                    continue
                
                # Check if c_in covers all critical neighbors of c_out
                if not all(u in neighbor_sets[c_in] for u in critical_nbs):
                    continue
                
                # Check if swap is valid:
                # For any u in neighbors[c_out] not in neighbors[c_in], counts[u] must be >= 2
                # (Because c_out is removed, so counts[u] decreases by 1. If it was 1, it becomes 0 -> invalid)
                # Note: If u is in critical_nbs, it is in neighbors[c_in], so counts[u] increases by 1 (net 0 change or +1 if c_in also covers it? No, c_in is added, so +1. c_out removed -1. Net 0. So it stays 1. This is fine as long as it's covered by c_in).
                # Wait, if u is in critical_nbs, counts[u] is 1.
                # Remove c_out: counts[u] becomes 0.
                # Add c_in: counts[u] becomes 1.
                # So it is covered.
                
                valid = True
                for u in neighbors[c_out]:
                    if u not in neighbor_sets[c_in]:
                        if counts[u] < 2:
                            valid = False
                            break
                
                if valid:
                    # Score: how many neighbors of c_out does c_in cover? (Higher overlap is better for stability?)
                    # Or: what is the new min count?
                    # Let's just collect a few valid swaps.
                    valid_candidates.append(c_in)
                    
                    if len(valid_candidates) > 10:
                        break

            if not valid_candidates:
                continue
            
            # Pick a random valid candidate
            c_in = random.choice(valid_candidates)
            
            # Apply swap
            for nb in neighbors[c_out]:
                counts[nb] -= 1
            for nb in neighbors[c_in]:
                counts[nb] += 1
            code_indices.remove(c_out)
            code_indices.add(c_in)
            
            found_swap = True
            
            # After swap, check for immediate removals
            # We don't break the loop immediately, we let the outer loop handle removals.
            # But breaking out of the c_out loop allows us to go back to removal phase.
            break

        if not found_swap:
            # No valid swap found. Perturb.
            # Pick a random c_out and a random c_in that makes a valid swap?
            # If no valid swap exists, we are stuck in a local minimum where no single swap helps.
            # We can try a random valid swap even if it doesn't obviously help, to escape.
            # Or, we can accept a "worse" swap? No, we must maintain validity.
            
            # Let's try to find ANY valid swap, even if it doesn't cover criticals?
            # No, validity requires covering criticals.
            
            # If no valid swap, we might be at the optimum for this configuration.
            # Try random perturbations:
            # Pick random c_out, random c_in.
            # If swap is valid, do it.
            # If not, skip.
            
            perturbed = False
            for _ in range(50):
                c_out = random.choice(list(code_indices))
                c_in = random.randint(0, total - 1)
                if c_in in code_indices:
                    continue
                
                critical_nbs = [u for u in neighbors[c_out] if counts[u] == 1]
                if not all(u in neighbor_sets[c_in] for u in critical_nbs):
                    continue
                
                valid = True
                for u in neighbors[c_out]:
                    if u not in neighbor_sets[c_in]:
                        if counts[u] < 2:
                            valid = False
                            break
                
                if valid:
                    for nb in neighbors[c_out]:
                        counts[nb] -= 1
                    for nb in neighbors[c_in]:
                        counts[nb] += 1
                    code_indices.remove(c_out)
                    code_indices.add(c_in)
                    perturbed = True
                    break
            
            if not perturbed:
                # Stuck. Reset to best found so far and try again with different random seed?
                # Or just break if we've been stuck for a while.
                # For now, let's just break if we can't find any move.
                # But we should try to restore best and restart?
                # Let's just break to save time, as we likely hit a local optimum.
                # However, we can try to "kick" by accepting a slightly worse state?
                # No, validity is strict.
                
                # If we can't find any valid swap, we are in a "rigid" configuration.
                # We might need to remove 2 and add 2? That's complex.
                # Let's just break.
                break

    # Convert best_code indices back to words
    return [list((best_code[i] // powers[p]) % q for p in range(n)) for i in range(len(best_code))]

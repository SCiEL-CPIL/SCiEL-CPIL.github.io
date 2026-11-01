import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # --- General fallbacks for non-target cases ---
    if not (q == 3 and n == 6 and radius == 1):
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # --- Specific solver for K_3(6, 1) ---
    N = q ** n  # 729
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute spheres as bitmasks for fast coverage operations
    # Sphere of a word w: all words within Hamming distance <= radius
    # For radius=1, sphere size is 1 + n*(q-1) = 1 + 6*2 = 13
    spheres = [0] * N
    sphere_lists = [[] for _ in range(N)]  # Also keep as lists for iteration
    
    for i, w in enumerate(all_words):
        mask = 1 << i
        sphere_lists[i].append(i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                j = word_to_idx[tuple(w)]
                mask |= (1 << j)
                sphere_lists[i].append(j)
            w[pos] = orig_val
        spheres[i] = mask
    
    full_mask = (1 << N) - 1

    # Precompute neighbor lists (same as spheres but as lists for quick iteration)
    neighbors = sphere_lists

    # --- Pruning: remove redundant codewords ---
    def prune(code_list):
        """Remove any codeword whose removal still leaves full coverage."""
        current = list(code_list)
        changed = True
        while changed:
            changed = False
            for i in range(len(current)):
                # Check if current[i] is redundant
                union_others = 0
                for j in range(len(current)):
                    if i == j:
                        continue
                    union_others |= spheres[current[j]]
                    if union_others == full_mask:
                        break
                if union_others == full_mask:
                    current.pop(i)
                    changed = True
                    break  # Restart the for loop
        return current

    # --- Greedy set cover with bitmask operations ---
    def greedy_construct(random_seed):
        random.seed(random_seed)
        candidates = list(range(N))
        random.shuffle(candidates)
        
        covered = 0
        chosen = []
        
        while covered != full_mask:
            best_gain = -1
            best_idx = -1
            
            for idx in candidates:
                if idx in chosen:
                    continue
                new_covered = spheres[idx] & ~covered
                gain = new_covered.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    
            if best_gain == 0:
                break
                
            chosen.append(best_idx)
            covered |= spheres[best_idx]
            
        return chosen

    # --- Local search: swap-based optimization ---
    def local_search(code_list, random_seed, time_budget):
        """Try to reduce code size via swaps and removals."""
        random.seed(random_seed)
        current = list(code_list)
        
        # Build coverage counts
        counts = np.zeros(N, dtype=np.int32)
        for c in current:
            for nb in neighbors[c]:
                counts[nb] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Remove redundant codewords
            to_remove = []
            for c in current:
                redundant = True
                for nb in neighbors[c]:
                    if counts[nb] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(c)
            
            if to_remove:
                for c in to_remove:
                    current.remove(c)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                improved = True
                continue
            
            # 2. Try swaps: replace one codeword with another to enable further removals
            if not current:
                break
                
            c_idx = random.randint(0, len(current) - 1)
            c = current[c_idx]
            
            # Find essential neighbors (count == 1)
            essential = [nb for nb in neighbors[c] if counts[nb] == 1]
            
            if not essential:
                # Should have been removed in step 1, but handle edge case
                continue
            
            # Try to find a replacement c' that covers all essential neighbors
            found_swap = False
            for _ in range(30):  # Try 30 random candidates
                if time.time() > deadline:
                    break
                
                c_prime = random.randint(0, N - 1)
                if c_prime in current:
                    continue
                
                # Check if c_prime covers all essential neighbors
                covers_all = True
                for nb in essential:
                    if nb not in neighbors[c_prime]:
                        covers_all = False
                        break
                
                if not covers_all:
                    continue
                
                # Check if swapping is valid (no word becomes uncovered)
                risk_words = [nb for nb in neighbors[c] if nb not in neighbors[c_prime]]
                
                valid = True
                for nb in risk_words:
                    if counts[nb] - 1 < 1:
                        valid = False
                        break
                
                if valid:
                    # Perform swap
                    current[c_idx] = c_prime
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                    for nb in neighbors[c_prime]:
                        counts[nb] += 1
                    found_swap = True
                    improved = True
                    break
            
            if not found_swap:
                # Try a perturbation: remove 1 random word and re-add greedily
                if len(current) > 1:
                    rem_idx = random.randint(0, len(current) - 1)
                    rem_c = current.pop(rem_idx)
                    for nb in neighbors[rem_c]:
                        counts[nb] -= 1
                    
                    # Re-add greedily
                    covered_mask = 0
                    for c2 in current:
                        covered_mask |= spheres[c2]
                    
                    best_gain = -1
                    best_idx = -1
                    for idx in range(N):
                        if idx in current:
                            continue
                        gain = (spheres[idx] & ~covered_mask).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = idx
                    
                    if best_idx != -1 and best_gain > 0:
                        current.append(best_idx)
                        for nb in neighbors[best_idx]:
                            counts[nb] += 1
                        improved = True
                    else:
                        # Revert
                        current.append(rem_c)
                        for nb in neighbors[rem_c]:
                            counts[nb] += 1
                else:
                    break

        return current

    # --- Branch and bound for small gaps ---
    def branch_and_bound(code_list, target_size, time_budget):
        """Try to find a covering of size <= target_size using B&B."""
        if len(code_list) <= target_size:
            return code_list
        
        # This is a simplified B&B; in practice, for 729 words,
        # we rely on the local search to get close to optimal.
        # For this problem, B&B is too expensive for the full space.
        # We use it only for verification or very small instances.
        return code_list

    # --- Main search loop ---
    best_code = None
    best_size = N  # Worst case

    # Start with tetracode direct sum (81 words) and prune
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = tuple(h + list(tail))
            init_code.append(word_to_idx[w])
    
    pruned_init = prune(init_code)
    if len(pruned_init) < best_size:
        best_size = len(pruned_init)
        best_code = pruned_init
    
    # Run greedy restarts + local search
    num_restarts = 300
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        seed = 1000 + r
        greedy_result = greedy_construct(seed)
        
        # Prune the greedy result
        pruned_greedy = prune(greedy_result)
        
        # Local search on the pruned result
        optimized = local_search(pruned_greedy, seed, deadline)
        
        if len(optimized) < best_size:
            best_size = len(optimized)
            best_code = optimized
            
            if best_size <= 73:
                # Found optimal or near-optimal; do a few more to be sure
                pass

    # If we haven't found 73, try more aggressive local search on best found
    if best_size > 73 and best_code is not None:
        # Try local search with different seeds on the best code
        for seed in range(10):
            if time.time() > deadline:
                break
            optimized = local_search(best_code, 5000 + seed, deadline)
            if len(optimized) < best_size:
                best_size = len(optimized)
                best_code = optimized

    if best_code is None:
        # Fallback
        best_code = prune(init_code)

    # Convert indices back to words
    result = [list(all_words[idx]) for idx in best_code]
    return result

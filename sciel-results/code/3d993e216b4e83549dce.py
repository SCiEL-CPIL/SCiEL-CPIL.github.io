"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = q ** n
    
    # Precompute neighbors using NumPy for speed
    words_array = np.array(all_words, dtype=np.int8)
    
    # Compute Hamming distances matrix
    diff_matrix = (words_array[:, np.newaxis, :] != words_array[np.newaxis, :, :]).sum(axis=2)
    
    # neighbors[i] = set of indices j where diff_matrix[i,j] <= radius
    neighbors = [set() for _ in range(num_words)]
    for i in range(num_words):
        mask = diff_matrix[i] <= radius
        neighbors[i] = set(np.where(mask)[0].tolist())
    
    # Greedy set-cover
    uncovered = set(range(num_words))
    chosen = []
    chosen_set = set()
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_candidate = -1
        best_count = 0
        
        for i in range(num_words):
            if i in chosen_set:
                continue
            count = 0
            for idx in neighbors[i]:
                if idx in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_candidate = i
        
        if best_candidate == -1 or best_count == 0:
            break
            
        chosen.append(best_candidate)
        chosen_set.add(best_candidate)
        uncovered -= neighbors[best_candidate]
    
    # Local search: try to remove codewords
    def try_remove(codeword_list, covered_set):
        """Try to remove each codeword if possible. Returns (new_list, new_covered, improved)."""
        new_list = list(codeword_list)
        new_covered = set(covered_set)
        improved = False
        i = 0
        while i < len(new_list):
            cw = new_list[i]
            # Check if any word is only covered by this codeword
            removable = True
            for w in neighbors[cw]:
                only_here = True
                for j, other in enumerate(new_list):
                    if j != i and w in neighbors[other]:
                        only_here = False
                        break
                if only_here:
                    removable = False
                    break
            if removable:
                new_list.pop(i)
                new_covered -= neighbors[cw]
                # Recompute covered from scratch
                new_covered = set()
                for cw2 in new_list:
                    new_covered |= neighbors[cw2]
                improved = True
                # Don't increment i, check next
            else:
                i += 1
        return new_list, new_covered, improved
    
    # Run local search multiple times
    for _ in range(5):
        if time.time() > deadline:
            break
        chosen, covered, improved = try_remove(chosen, set().union(*[neighbors[c] for c in chosen]) if chosen else set())
        if not improved:
            break
    
    # Simulated annealing
    if time.time() < deadline:
        sa_deadline = deadline
        
        def build_covered(codeword_indices):
            if not codeword_indices:
                return set()
            covered = set()
            for i in codeword_indices:
                covered |= neighbors[i]
            return covered
        
        current = list(chosen)
        current_covered = build_covered(current)
        best = list(current)
        best_covered = set(current_covered)
        best_count = len(best)
        
        T = 3.0
        T_min = 0.01
        cooling = 0.999
        
        while T > T_min and time.time() < sa_deadline:
            if time.time() > sa_deadline - 2:
                break
                
            move_type = random.random()
            
            if move_type < 0.6 and len(current) > 1:
                # Remove a random codeword
                idx_to_remove = random.choice(current)
                
                # Count words only covered by this codeword
                newly_uncovered = 0
                for w in neighbors[idx_to_remove]:
                    only_covered = True
                    for other in current:
                        if other != idx_to_remove and w in neighbors[other]:
                            only_covered = False
                            break
                    if only_covered:
                        newly_uncovered += 1
                
                if newly_uncovered == 0:
                    current.remove(idx_to_remove)
                    current_covered = build_covered(current)
                    if len(current) < best_count:
                        best = list(current)
                        best_covered = set(current_covered)
                        best_count = len(best)
                else:
                    if random.random() < np.exp(-newly_uncovered / T):
                        current.remove(idx_to_remove)
                        current_covered = build_covered(current)
                        if len(current) < best_count:
                            best = list(current)
                            best_covered = set(current_covered)
                            best_count = len(best)
                        
            elif move_type < 0.95 and len(current) > 0:
                # Swap
                idx_to_remove = random.choice(current)
                current_set = set(current)
                candidates = [i for i in range(num_words) if i not in current_set]
                if not candidates:
                    continue
                
                # Find words only covered by idx_to_remove
                words_to_cover = set()
                for w in neighbors[idx_to_remove]:
                    only_covered = True
                    for other in current:
                        if other != idx_to_remove and w in neighbors[other]:
                            only_covered = False
                            break
                    if only_covered:
                        words_to_cover.add(w)
                
                if words_to_cover:
                    # Score candidates
                    best_add = None
                    best_score = -1
                    for cand in candidates:
                        score = len(neighbors[cand] & words_to_cover)
                        if score > best_score:
                            best_score = score
                            best_add = cand
                    if best_score == 0:
                        best_add = random.choice(candidates)
                else:
                    best_add = random.choice(candidates)
                
                current.remove(idx_to_remove)
                current.append(best_add)
                current_covered = build_covered(current)
                
                if len(current_covered) == num_words:
                    if len(current) < best_count:
                        best = list(current)
                        best_covered = set(current_covered)
                        best_count = len(best)
                else:
                    penalty = num_words - len(current_covered)
                    if random.random() < np.exp(-penalty / T):
                        pass
                    else:
                        current.remove(best_add)
                        current.append(idx_to_remove)
                        current_covered = build_covered(current)
                        
            else:
                # Add
                current_set = set(current)
                candidates = [i for i in range(num_words) if i not in current_set]
                if not candidates:
                    continue
                idx_to_add = random.choice(candidates)
                current.append(idx_to_add)
                current_covered = build_covered(current)
                
                if len(current_covered) < num_words:
                    if random.random() < np.exp(-0.3 / T):
                        pass
                    else:
                        current.remove(idx_to_add)
                        current_covered = build_covered(current)
                else:
                    current.remove(idx_to_add)
                    current_covered = build_covered(current)
                    
            T *= cooling
        
        if len(best) < len(chosen):
            chosen = best
    
    # Final local search
    for _ in range(3):
        if time.time() > deadline:
            break
        chosen, covered, improved = try_remove(chosen, set().union(*[neighbors[c] for c in chosen]) if chosen else set())
        if not improved:
            break
    
    # Convert indices back to word tuples/lists
    result = [list(all_words[i]) for i in chosen]
    
    return result
# EVOLVE-BLOCK-END

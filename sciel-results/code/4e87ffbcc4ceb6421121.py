import time
import random
import itertools
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    if n <= 1:
        return "".join(str(i) for i in range(1, n + 1))
    
    word = [1]
    
    for m in range(2, n + 1):
        # Extract distinct (m-1) permutations from word
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        N = len(perms)
        
        # For small N, use simple ordering
        if N <= 5:
            ordered_perms = perms
        else:
            # Check if we have enough time for optimization
            if time.time() > deadline - 30:
                ordered_perms = perms
            else:
                # Compute overlap matrix using numpy for speed
                # Convert perms to numpy array for faster slicing
                perms_np = np.array(perms)
                
                # Precompute suffixes and prefixes as tuples for quick lookup
                # Actually, let's use a different approach:
                # For each permutation, store a dictionary mapping suffix tuples to the permutation index
                # and prefix tuples to the permutation index
                
                # Compute overlap matrix
                max_ov = m - 1
                overlap_matrix = np.zeros((N, N), dtype=np.int8)
                
                for i in range(N):
                    pi = perms_np[i]
                    for j in range(N):
                        pj = perms_np[j]
                        best = 0
                        for k in range(max_ov, 0, -1):
                            if np.array_equal(pi[-k:], pj[:k]):
                                best = k
                                break
                        overlap_matrix[i][ j] = best
                
                # Greedy path with multiple restarts and soft-max selection
                time_remaining = deadline - time.time() - 10
                if time_remaining < 5:
                    n_restarts = 10
                elif time_remaining < 15:
                    n_restarts = 50
                elif time_remaining < 30:
                    n_restarts = 100
                else:
                    n_restarts = 200
                
                best_path = None
                best_score = -1
                temperature = 0.1  # Soft-max temperature
                
                for r in range(n_restarts):
                    start = random.randint(0, N - 1)
                    path = [start]
                    visited = np.zeros(N, dtype=bool)
                    visited[start] = True
                    current = start
                    current_score = 0.0
                    
                    for _ in range(N - 1):
                        # Get overlaps to unvisited nodes
                        overlaps = overlap_matrix[current].copy()
                        overlaps[visited] = -100  # Penalize visited nodes
                        
                        # Use soft-max to select next node with some randomness
                        # This allows exploration of the solution space
                        if temperature > 0:
                            # Soft-max: probability proportional to exp(overlap / temperature)
                            # Shift to avoid overflow
                            max_ov_val = np.max(overlaps)
                            if max_ov_val > -100:
                                exp_vals = np.exp((overlaps - max_ov_val) / temperature)
                                total = np.sum(exp_vals)
                                if total > 0:
                                    probs = exp_vals / total
                                    # Sample from distribution
                                    best_next = np.random.choice(N, p=probs)
                                else:
                                    best_next = np.argmax(overlaps)
                            else:
                                best_next = np.argmax(overlaps)
                        else:
                            best_next = np.argmax(overlaps)
                        
                        visited[best_next] = True
                        path.append(best_next)
                        current_score += overlap_matrix[current, best_next]
                        current = best_next
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_path = path
                    
                    # Check time
                    if time.time() > deadline - 15:
                        break
                
                if best_path is not None:
                    ordered_perms = [perms[i] for i in best_path]
                else:
                    ordered_perms = perms
        
        # Build the new word
        new = []
        for idx in range(len(ordered_perms)):
            p = ordered_perms[idx]
            block = list(p) + [m] + list(p)
            
            if idx == 0:
                new += block
            else:
                # Find maximum overlap between end of new and start of block
                max_ov = min(len(new), len(block))
                ov = 0
                for k in range(max_ov, 0, -1):
                    if new[-k:] == block[:k]:
                        ov = k
                        break
                new += block[ov:]
        
        word = new
        
        if time.time() > deadline:
            break
    
    return "".join(map(str, word))

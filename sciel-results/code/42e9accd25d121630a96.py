import time
import numpy as np
import random
from collections import Counter

def construct():
    n = 16
    num_edges = n * (n - 1) // 2
    # Precompute edges and K4s
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    k4s = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    e1 = edge_index[(a, b)]
                    e2 = edge_index[(a, c)]
                    e3 = edge_index[(a, d)]
                    e4 = edge_index[(b, c)]
                    e5 = edge_index[(b, d)]
                    e6 = edge_index[(c, d)]
                    k4s.append((e1, e2, e3, e4, e5, e6))
    
    # Map edge to K4s containing it
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, k4 in enumerate(k4s):
        for e in k4:
            edge_to_k4s[e].append(k4_idx)

    def is_valid_k4(colors, k4_idx):
        c = [colors[e] for e in k4s[k4_idx]]
        return len(set(c)) >= 5

    def count_violations(colors):
        v = 0
        for k4_idx in range(len(k4s)):
            if not is_valid_k4(colors, k4_idx):
                v += 1
        return v

    def evaluate_delta(colors, e, new_c, current_violations_set):
        # current_violations_set is a set of k4_idx that are currently violating
        delta = 0
        for k4_idx in edge_to_k4s[e]:
            # Get colors for this K4
            k4 = k4s[k4_idx]
            old_colors = [colors[ei] for ei in k4]
            # Apply change
            old_c = colors[e]
            if old_c == new_c:
                continue
            
            # Find index of e in k4
            idx_in_k4 = k4.index(e)
            old_colors[idx_in_k4] = new_c
            
            old_valid = (k4_idx not in current_violations_set)
            new_valid = (len(set(old_colors)) >= 5)
            
            if old_valid and not new_valid:
                delta += 1
            elif not old_valid and new_valid:
                delta -= 1
        return delta

    deadline = time.time() + 110
    
    # Try K = 13, 14, 15
    for K in [13, 14, 15]:
        # Multiple restarts
        for restart in range(1000):
            if time.time() > deadline:
                break
            
            colors = np.random.randint(0, K, num_edges)
            # Convert to list for faster access in inner loop? Or keep as list
            colors_list = colors.tolist()
            
            # Initial violations
            viol_set = set()
            for k4_idx in range(len(k4s)):
                if not is_valid_k4(colors_list, k4_idx):
                    viol_set.add(k4_idx)
            
            if not viol_set:
                return colors_list
            
            # Local search
            for step in range(10000):
                if time.time() > deadline:
                    break
                if not viol_set:
                    return colors_list
                
                # Pick random violated K4
                k4_idx = random.choice(list(viol_set))
                k4 = k4s[k4_idx]
                
                best_delta = 0
                best_e = -1
                best_c = -1
                
                # Try changing each edge in this K4
                for e in k4:
                    for new_c in range(K):
                        if new_c == colors_list[e]:
                            continue
                        d = evaluate_delta(colors_list, e, new_c, viol_set)
                        if d < best_delta:
                            best_delta = d
                            best_e = e
                            best_c = new_c
                
                if best_e != -1:
                    # Apply best move
                    old_c = colors_list[best_e]
                    colors_list[best_e] = best_c
                    
                    # Update viol_set
                    for k4_idx_affected in edge_to_k4s[best_e]:
                        # Re-evaluate
                        if is_valid_k4(colors_list, k4_idx_affected):
                            if k4_idx_affected in viol_set:
                                viol_set.remove(k4_idx_affected)
                        else:
                            if k4_idx_affected not in viol_set:
                                viol_set.add(k4_idx_affected)
                else:
                    # No improving move, break or randomize
                    # Simple randomization
                    e = random.choice(k4)
                    new_c = random.randint(0, K - 1)
                    if new_c != colors_list[e]:
                        colors_list[e] = new_c
                        for k4_idx_affected in edge_to_k4s[e]:
                            if is_valid_k4(colors_list, k4_idx_affected):
                                if k4_idx_affected in viol_set:
                                    viol_set.remove(k4_idx_affected)
                            else:
                                if k4_idx_affected not in viol_set:
                                    viol_set.add(k4_idx_affected)
            
            if time.time() > deadline:
                break

    # Fallback to seed
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

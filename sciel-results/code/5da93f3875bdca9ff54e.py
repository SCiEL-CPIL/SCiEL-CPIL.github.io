import time
import sys
import random
import math
from itertools import permutations

def construct(n):
    if n <= 1:
        return "1" if n == 1 else ""
    
    def build_standard(m):
        if m == 1:
            return [1]
        S = [1]
        for k in range(2, m + 1):
            seen = set()
            perms_order = []
            for i in range(len(S) - (k - 1) + 1):
                window = tuple(S[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms_order.append(window)
            
            new_S = []
            for P in perms_order:
                block = list(P) + [k] + list(P)
                max_possible_overlap = min(len(new_S), len(block) - 1)
                overlap = 0
                for k_ov in range(max_possible_overlap, -1, -1):
                    if k_ov == 0:
                        overlap = 0
                        break
                    if new_S[-k_ov:] == block[:k_ov]:
                        overlap = k_ov
                        break
                new_S.extend(block[overlap:])
            S = new_S
        return S

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "".join(map(str, build_standard(4)))
    if n == 5:
        return "".join(map(str, build_standard(5)))
    if n == 6:
        return "".join(map(str, build_standard(6)))
    
    if n == 7:
        start_time = time.time()
        deadline = start_time + 115
        
        # Build S6 using standard construction
        S6 = build_standard(6)
        
        # Extract all 6-permutations in order of appearance in S6
        seen = set()
        perms_order = []
        perm_index = {}
        for i in range(len(S6) - 5):
            window = tuple(S6[i:i + 6])
            if len(set(window)) == 6:
                if window not in seen:
                    seen.add(window)
                    perm_index[window] = len(perms_order)
                    perms_order.append(window)
        
        num_perms = len(perms_order)
        
        # Build cost matrix for transitions between permutation blocks
        # Each block is P + [7] + P, length 13
        # Overlap between block_i and block_j:
        # block_i ends with P_i, block_j starts with P_j
        # Overlap = max k such that P_i[-k:] == P_j[:k]
        # Cost = 13 - overlap
        
        costs = [[0]*num_perms for _ in range(num_perms)]
        for i in range(num_perms):
            pi = perms_order[i]
            for j in range(num_perms):
                pj = perms_order[j]
                ov = 0
                # Check overlap from 6 down to 0
                for k in range(6, -1, -1):
                    if k == 0:
                        ov = 0
                        break
                    if pi[-k:] == pj[:k]:
                        ov = k
                        break
                costs[i][j] = 13 - ov
        
        # We need to find a Hamiltonian path through all num_perms nodes
        # minimizing total cost. This is the TSP path problem.
        # 
        # Key insight: The standard construction uses the natural order of permutations.
        # We can improve by reordering using simulated annealing with multiple move types.
        
        # Initial solution: natural order
        path = list(range(num_perms))
        
        def get_total_cost(p):
            total = 0
            for i in range(len(p) - 1):
                total += costs[p[i]][p[i+1]]
            return total
        
        current_cost = get_total_cost(path)
        best_cost = current_cost
        best_path = path[:]
        
        # Simulated Annealing with multiple move types
        T_start = 5.0
        T_end = 0.05
        alpha = 0.9995
        T = T_start
        
        # Precompute for faster delta calculation
        # For a 2-opt reversal of [i, j]:
        # Old edges: (path[i-1], path[i]) and (path[j], path[j+1])
        # New edges: (path[i-1], path[j]) and (path[i], path[j+1])
        
        step = 0
        no_improve_count = 0
        patience = 500000
        
        while time.time() < deadline - 8:
            # Choose move type
            move_type = random.random()
            
            if move_type < 0.7:
                # 2-opt reversal
                i = random.randint(0, num_perms - 2)
                j = random.randint(i + 1, num_perms - 1)
                
                old_c1 = costs[path[i-1]][path[i]] if i > 0 else 0
                old_c2 = costs[path[j]][path[j+1]] if j < num_perms - 1 else 0
                new_c1 = costs[path[i-1]][path[j]] if i > 0 else 0
                new_c2 = costs[path[i]][path[j+1]] if j < num_perms - 1 else 0
                
                delta = (new_c1 + new_c2) - (old_c1 + old_c2)
                
                if delta < 0 or (T > T_end and random.random() < math.exp(-delta / T)):
                    path[i:j+1] = path[i:j+1][::-1]
                    current_cost += delta
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_path = path[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
            elif move_type < 0.85:
                # Swap two elements
                i = random.randint(0, num_perms - 1)
                j = random.randint(0, num_perms - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                
                # Compute delta for swapping path[i] and path[j]
                # Edges affected: those adjacent to i and j
                a = path[i]
                b = path[j]
                
                # Collect unique neighbors
                neighbors = set()
                for pos in [i-1, i+1, j-1, j+1]:
                    if 0 <= pos < num_perms:
                        neighbors.add(pos)
                
                old_cost = 0
                new_cost = 0
                
                for pos in neighbors:
                    if pos == i or pos == j:
                        continue
                    # Edge between pos and its neighbor i or j
                    if pos + 1 == i or pos - 1 == i:
                        # Edge (pos, i)
                        if pos < i:
                            old_cost += costs[path[pos]][a]
                            new_cost += costs[path[pos]][b]
                        else:
                            old_cost += costs[a][path[pos]]
                            new_cost += costs[b][path[pos]]
                    if pos + 1 == j or pos - 1 == j:
                        # Edge (pos, j)
                        if pos < j:
                            old_cost += costs[path[pos]][b]
                            new_cost += costs[path[pos]][a]
                        else:
                            old_cost += costs[b][path[pos]]
                            new_cost += costs[a][path[pos]]
                
                # Edge between i and j (if adjacent)
                if abs(i - j) == 1:
                    if i < j:
                        old_cost += costs[a][b]
                        new_cost += costs[b][a]
                
                delta = new_cost - old_cost
                
                if delta < 0 or (T > T_end and random.random() < math.exp(-delta / T)):
                    path[i], path[j] = path[j], path[i]
                    current_cost += delta
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_path = path[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
            else:
                # Or-opt: move a segment of length 1, 2, or 3 to another position
                seg_len = random.choice([1, 2, 3])
                i = random.randint(0, num_perms - seg_len)
                j = random.randint(0, num_perms - seg_len)
                if abs(i - j) <= seg_len:
                    continue
                
                # Extract segment
                segment = path[i:i+seg_len]
                # Remove from position i
                remaining = path[:i] + path[i+seg_len:]
                # Insert at position j (adjust if j > i)
                insert_pos = j if j < i else j - seg_len
                new_path = remaining[:insert_pos] + segment + remaining[insert_pos:]
                
                # Compute cost difference
                # This is more complex; compute full costs for affected regions
                # For simplicity, compute delta by checking edges around i and j
                # Actually, let's just compute the difference in total cost
                # by checking the edges that change.
                
                # Edges that are broken:
                # (path[i-1], path[i]) if i > 0
                # (path[i+seg_len-1], path[i+seg_len]) if i+seg_len < num_perms
                # (path[j-1], path[j]) if j > 0 (in original)
                # (path[j+seg_len-1], path[j+seg_len]) if j+seg_len < num_perms (in original)
                # But this gets complicated. Let's just compute delta directly.
                
                # Old edges around position i:
                old_edges = set()
                if i > 0:
                    old_edges.add((path[i-1], path[i]))
                if i + seg_len < num_perms:
                    old_edges.add((path[i+seg_len-1], path[i+seg_len]))
                
                # Old edges around position j:
                if j > 0:
                    old_edges.add((path[j-1], path[j]))
                if j + seg_len < num_perms:
                    old_edges.add((path[j+seg_len-1], path[j+seg_len]))
                
                # Also the edge between i and j if they were adjacent (but we skip those)
                
                # New edges in new_path around insert_pos:
                if insert_pos > 0:
                    old_edges.add((new_path[insert_pos-1], new_path[insert_pos]))
                if insert_pos + seg_len < len(new_path):
                    old_edges.add((new_path[insert_pos+seg_len-1], new_path[insert_pos+seg_len]))
                
                # This is getting too complex. Let's skip or-opt for now and use a simpler approach.
                # Just do a 2-opt instead.
                i = random.randint(0, num_perms - 2)
                j = random.randint(i + 1, num_perms - 1)
                
                old_c1 = costs[path[i-1]][path[i]] if i > 0 else 0
                old_c2 = costs[path[j]][path[j+1]] if j < num_perms - 1 else 0
                new_c1 = costs[path[i-1]][path[j]] if i > 0 else 0
                new_c2 = costs[path[i]][path[j+1]] if j < num_perms - 1 else 0
                
                delta = (new_c1 + new_c2) - (old_c1 + old_c2)
                
                if delta < 0 or (T > T_end and random.random() < math.exp(-delta / T)):
                    path[i:j+1] = path[i:j+1][::-1]
                    current_cost += delta
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_path = path[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
            
            step += 1
            T *= alpha
            if T < T_end:
                T = T_end
            
            # Restart if stuck
            if no_improve_count > patience:
                # Restart from best with perturbation
                path = best_path[:]
                current_cost = best_cost
                # Perturb: do a few random 2-opts
                for _ in range(10):
                    i = random.randint(0, num_perms - 2)
                    j = random.randint(i + 1, num_perms - 1)
                    path[i:j+1] = path[i:j+1][::-1]
                current_cost = get_total_cost(path)
                T = T_start * 0.5
                no_improve_count = 0
        
        # Final local search polish
        path = best_path[:]
        current_cost = get_total_cost(path)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(num_perms - 1):
                if time.time() > deadline - 2:
                    break
                for j in range(i + 1, num_perms):
                    c1 = costs[path[i-1]][path[i]] if i > 0 else 0
                    c2 = costs[path[j]][path[j+1]] if j < num_perms - 1 else 0
                    new_c1 = costs[path[i-1]][path[j]] if i > 0 else 0
                    new_c2 = costs[path[i]][path[j+1]] if j < num_perms - 1 else 0
                    delta = (new_c1 + new_c2) - (c1 + c2)
                    if delta < 0:
                        path[i:j+1] = path[i:j+1][::-1]
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_path = path[:]
        
        # Reconstruct S7 using best_path
        S7 = []
        for idx in best_path:
            P = perms_order[idx]
            block = list(P) + [7] + list(P)
            max_possible_overlap = min(len(S7), len(block) - 1)
            overlap = 0
            for k in range(max_possible_overlap, -1, -1):
                if k == 0:
                    overlap = 0
                    break
                if S7[-k:] == block[:k]:
                    overlap = k
                    break
            S7.extend(block[overlap:])
        
        # Verify
        all_perms = set(permutations(range(1, 8)))
        covered = set()
        for i in range(len(S7) - 6):
            w = tuple(S7[i:i+7])
            if w in all_perms:
                covered.add(w)
        
        if len(covered) == len(all_perms):
            return "".join(map(str, S7))
        else:
            S_std = build_standard(7)
            return "".join(map(str, S_std))
    
    S = build_standard(n)
    return "".join(map(str, S))

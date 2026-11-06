"""GRASP (Greedy Randomized Adaptive Search Procedure) for 3-AP maximization."""
import random
import time

def construct():
    n = 256
    random.seed(42)
    
    # Precompute APs
    three_aps = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
            
    four_aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    # Maps for incremental updates
    elem_to_3aps = {i: [] for i in range(1, n + 1)}
    for idx, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3aps[x].append(idx)
            
    elem_to_4aps = {i: [] for i in range(1, n + 1)}
    for idx, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(idx)

    def local_search(S):
        """Improve S via swaps and additions."""
        S_set = set(S)
        in_set = [False] * (n + 1)
        for x in S_set:
            in_set[x] = True
            
        # Initial counts
        count_3 = 0
        for ap in three_aps:
            if in_set[ap[0]] and in_set[ap[1]] and in_set[ap[2]]:
                count_3 += 1
        count_4 = 0
        for ap in four_aps:
            if in_set[ap[0]] and in_set[ap[1]] and in_set[ap[2]] and in_set[ap[3]]:
                count_4 += 1
                
        if count_4 > 0:
            return S_set # Should not happen if construction is valid
            
        current_count = count_3
        improved = True
        while improved:
            improved = False
            # Try swaps
            for a in list(S_set):
                for b in range(1, n + 1):
                    if b in S_set:
                        continue
                    # Check if swap (remove a, add b) is valid
                    # Remove a
                    in_set[a] = False
                    # Add b
                    in_set[b] = True
                    
                    # Check 4-APs involving a or b
                    # We need to check if any 4-AP becomes fully present.
                    # Since we removed a, no 4-AP containing a can be fully present.
                    # 4-APs containing b might become fully present.
                    valid = True
                    for idx in elem_to_4aps[b]:
                        ap = four_aps[idx]
                        if all(in_set[x] for x in ap):
                            valid = False
                            break
                            
                    if not valid:
                        in_set[b] = False
                        in_set[a] = True
                        continue
                        
                    # Calculate new count_3
                    # Remove 3-APs involving a
                    delta_3 = 0
                    for idx in elem_to_3aps[a]:
                        ap = three_aps[idx]
                        others = [y for y in ap if y != a]
                        if all(in_set[y] for y in others):
                            delta_3 -= 1
                            
                    # Add 3-APs involving b
                    for idx in elem_to_3aps[b]:
                        ap = three_aps[idx]
                        others = [y for y in ap if y != b]
                        if all(in_set[y] for y in others):
                            delta_3 += 1
                            
                    new_count = current_count + delta_3
                    if new_count > current_count:
                        S_set.remove(a)
                        S_set.add(b)
                        current_count = new_count
                        improved = True
                        break
                    else:
                        # Undo
                        in_set[b] = False
                        in_set[a] = True
                if improved:
                    break
                    
            # Try additions
            if not improved:
                for b in range(1, n + 1):
                    if b in S_set:
                        continue
                    # Check validity
                    in_set[b] = True
                    valid = True
                    for idx in elem_to_4aps[b]:
                        ap = four_aps[idx]
                        if all(in_set[x] for x in ap):
                            valid = False
                            break
                    if not valid:
                        in_set[b] = False
                        continue
                        
                    delta_3 = 0
                    for idx in elem_to_3aps[b]:
                        ap = three_aps[idx]
                        others = [y for y in ap if y != b]
                        if all(in_set[y] for y in others):
                            delta_3 += 1
                            
                    new_count = current_count + delta_3
                    if new_count > current_count:
                        S_set.add(b)
                        current_count = new_count
                        improved = True
                        break
                    else:
                        in_set[b] = False
        return S_set

    best_S = set()
    best_count = 0
    start_time = time.time()
    max_time = 55
    
    num_iterations = 30
    alpha = 0.2
    
    for _ in range(num_iterations):
        if time.time() - start_time > max_time:
            break
            
        S = set()
        remaining = set(range(1, n + 1))
        
        while remaining:
            candidates = []
            for x in remaining:
                # Check feasibility
                # We can't easily check has_4ap(S | {x}) without adding x.
                # But we can check if x creates a 4-AP with S.
                creates_4ap = False
                for idx in elem_to_4aps[x]:
                    ap = four_aps[idx]
                    others = [y for y in ap if y != x]
                    if all(y in S for y in others):
                        creates_4ap = True
                        break
                if creates_4ap:
                    continue
                
                # Calculate gain
                gain = 0
                for idx in elem_to_3aps[x]:
                    ap = three_aps[idx]
                    others = [y for y in ap if y != x]
                    if all(y in S for y in others):
                        gain += 1
                candidates.append((gain, x))
                
            if not candidates:
                break
                
            candidates.sort(reverse=True)
            k = max(1, int(len(candidates) * alpha))
            top_k = candidates[:k]
            _, x = random.choice(top_k)
            
            S.add(x)
            remaining.remove(x)
            
        S = local_search(S)
        # Count 3-APs for best_S update
        count = 0
        S_set = set(S)
        for ap in three_aps:
            if ap[0] in S_set and ap[1] in S_set and ap[2] in S_set:
                count += 1
                
        if count > best_count:
            best_count = count
            best_S = S

    return sorted(best_S)

import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    L = list(range(8))
    R = list(range(8, 16))
    
    # Precompute 4-subsets of L and R
    subsets_L = list(combinations(L, 4))
    subsets_R = list(combinations(R, 4))
    
    # For each pair of subsets, we need to check if the induced bipartite graph is Q3.
    # Q3 is 3-regular bipartite on 4+4.
    # So we just check degrees.
    
    def is_q3_aligned(mask, sL, sR):
        # mask is 64-bit int.
        # sL, sR are tuples of 4 vertices.
        # Check if subgraph induced by sL, sR is 3-regular.
        # Local indices: sL[i] -> row i, sR[j] -> col j.
        # Edge exists if bit (sL[i] - 0) * 8 + (sR[j] - 8) is set?
        # Let's use a simpler bit mapping.
        # Bit index = (u - 0) * 8 + (v - 8) for u in L, v in R.
        # u in 0..7, v in 8..15.
        # idx = u * 8 + (v - 8)
        
        # Check degrees
        # For each u in sL, count neighbors in sR
        # For each v in sR, count neighbors in sL
        
        # Unpack sL, sR to local 0..3 indices
        # u_local = i, v_local = j
        # Bit index = sL[i] * 8 + (sR[j] - 8)
        
        deg_L = [0] * 4
        deg_R = [0] * 4
        
        for i in range(4):
            u = sL[i]
            for j in range(4):
                v = sR[j]
                bit_idx = u * 8 + (v - 8)
                if (mask >> bit_idx) & 1:
                    deg_L[i] += 1
                    deg_R[j] += 1
        
        # Check if all degrees are 3
        for d in deg_L:
            if d != 3:
                return False
        for d in deg_R:
            if d != 3:
                return False
        return True

    def count_q3s(mask):
        count = 0
        for sL in subsets_L:
            for sR in subsets_R:
                if is_q3_aligned(mask, sL, sR):
                    count += 1
        return count

    def has_q3(mask):
        for sL in subsets_L:
            for sR in subsets_R:
                if is_q3_aligned(mask, sL, sR):
                    return True, sL, sR
        return False, None, None

    # Initial random graph
    random.seed(42)
    best_mask = 0
    best_count = 0
    
    for attempt in range(100):
        if time.time() > deadline - 30:
            break
            
        # Generate random mask with density 0.8
        mask = 0
        for i in range(64):
            if random.random() < 0.8:
                mask |= (1 << i)
                
        # Remove edges to make Q3-free
        changed = True
        while changed:
            changed = False
            has, sL, sR = has_q3(mask)
            if has:
                # Remove a random edge from this Q3
                # Find an edge in the Q3
                # We know it's 3-regular, so pick any edge.
                # Let's pick the first one.
                for i in range(4):
                    u = sL[i]
                    for j in range(4):
                        v = sR[j]
                        bit_idx = u * 8 + (v - 8)
                        if (mask >> bit_idx) & 1:
                            mask &= ~(1 << bit_idx)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break # Restart check loop as mask changed
            
        count = bin(mask).count('1')
        if count > best_count:
            best_count = count
            best_mask = mask
            
        # Greedy add
        for _ in range(1000):
            if time.time() > deadline - 30:
                break
            # Try to add a random missing edge
            missing_edges = []
            for i in range(64):
                if not (best_mask >> i) & 1:
                    missing_edges.append(i)
            if not missing_edges:
                break
            idx = random.choice(missing_edges)
            new_mask = best_mask | (1 << idx)
            has, _, _ = has_q3(new_mask)
            if not has:
                best_mask = new_mask
                best_count += 1
                
    # Simulated Annealing
    current_mask = best_mask
    current_count = best_count
    temperature = 1.0
    
    while time.time() < deadline - 5:
        # Flip a random bit
        idx = random.randint(0, 63)
        new_mask = current_mask ^ (1 << idx)
        
        # Check if new_mask is Q3-free
        has, _, _ = has_q3(new_mask)
        
        if not has:
            new_count = bin(new_mask).count('1')
            delta = new_count - current_count
            if delta > 0 or random.random() < 1.0 / (1.0 + delta * temperature):
                current_mask = new_mask
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_mask = current_mask
        else:
            # If it creates a Q3, we reject (or could try to repair, but simple reject is safer)
            pass
            
        temperature *= 0.995
        if temperature < 0.01:
            temperature = 0.01

    # Convert mask to edge list
    edges = []
    for i in range(64):
        if (best_mask >> i) & 1:
            u = i // 8
            v = (i % 8) + 8
            edges.append([u, v])
            
    return edges

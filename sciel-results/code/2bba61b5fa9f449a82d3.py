import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer for safety

    best_edge_count = float('inf')
    best_edges = []
    best_n = 0

    # We'll try different n values.
    # n=8: C(8,5)=56 edges.
    # n=9: C(9,5)=126 edges.
    # n=10: C(10,5)=252 edges.
    # n=11: C(11,5)=462 edges.
    # n=12: C(12,5)=792 edges.
    
    # The known 51-edge solution is likely for n=9 or n=10.
    # We will focus on n=8, 9, 10, 11.
    
    for n in range(8, 12):
        if time.time() > deadline:
            break
            
        all_edges = [list(e) for e in combinations(range(n), 5)]
        num_edges = len(all_edges)
        
        if num_edges > 200: # Constraint: at most 200 edges
            continue
            
        # Precompute kill masks
        # kill_mask[i] is a bitmask of length 2^n.
        # Bit c is set if coloring c (as integer 0..2^n-1) is monochromatic on edge i.
        kill_masks = [0] * num_edges
        
        for i, edge in enumerate(all_edges):
            mask_e = 0
            for v in edge:
                mask_e |= (1 << v)
            
            km = 0
            for c in range(1 << n):
                # Check if coloring c is monochromatic on edge mask_e
                # c & mask_e == 0 (all 0) or c & mask_e == mask_e (all 1)
                inter = c & mask_e
                if inter == 0 or inter == mask_e:
                    km |= (1 << c)
            kill_masks[i] = km

        # Greedy Max Coverage
        # We will run multiple iterations with randomization to find a better set
        # The greedy choice is deterministic if we break ties by index.
        # To explore different solutions, we can randomize the order of edges considered
        # or add small random noise to the coverage score.
        
        num_runs = 20 # Adjust based on time
        
        for run in range(num_runs):
            if time.time() > deadline:
                break
                
            survivors = (1 << (1 << n)) - 1
            used_edges = []
            remaining_edges = list(range(num_edges))
            random.shuffle(remaining_edges) # Randomize order to break ties differently
            
            while survivors != 0:
                if time.time() > deadline:
                    break
                
                best_edge_idx = -1
                best_score = -1
                best_kill = 0
                
                # Iterate through remaining edges
                for idx in remaining_edges:
                    # Score is number of survivors killed
                    # We can optimize this by only checking edges that have significant overlap
                    # But with small N, full check is fine.
                    score = bin(survivors & kill_masks[idx]).count('1')
                    
                    # Add small random noise to break ties and explore
                    if run > 0:
                        score += random.random() * 0.1
                        
                    if score > best_score:
                        best_score = score
                        best_edge_idx = idx
                        best_kill = kill_masks[idx]
                        
                if best_edge_idx == -1:
                    # No edge can kill any survivors? Should not happen if survivors != 0
                    # unless we ran out of edges, but we have all edges available.
                    break
                
                used_edges.append(all_edges[best_edge_idx])
                survivors &= ~best_kill
                
                # Remove used edge from remaining to avoid re-adding (though not strictly necessary for correctness, 
                # it prevents duplicate edges in the final set if we just track indices)
                # Actually, we should not add the same edge twice. 
                # Since we are building a set of distinct edges, we should remove it from candidates.
                try:
                    remaining_edges.remove(best_edge_idx)
                except ValueError:
                    pass
                
            if survivors == 0 and len(used_edges) < best_edge_count:
                best_edge_count = len(used_edges)
                best_edges = used_edges
                best_n = n
                
                # Early exit if we reach the theoretical lower bound or known best
                if best_edge_count <= 51:
                    # We can try to run more to see if we can get lower, 
                    # but 51 is the known best. 
                    # Let's continue briefly to see if 50 is possible, but break if time is tight.
                    if time.time() > deadline - 5:
                        break

    # Final verification and formatting
    # The task requires distinct edges. Our algorithm ensures distinct edges by removing from candidates.
    # We need to ensure n is between 5 and 26.
    if best_n == 0:
        # Fallback if something went wrong (should not happen)
        # Return a known valid instance if possible, or just return empty? 
        # Better to return a valid one from a quick run.
        # Let's just return a simple K_9^5 subset or similar if we failed.
        # But the greedy loop should always succeed.
        n = 9
        all_edges = [list(e) for e in combinations(range(n), 5)]
        return {'n': n, 'edges': all_edges}

    return {'n': best_n, 'edges': best_edges}

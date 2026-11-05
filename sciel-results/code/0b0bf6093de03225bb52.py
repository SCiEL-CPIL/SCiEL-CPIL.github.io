import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    def is_2_colorable_fast(n, edges, kill_masks_by_edge):
        """
        Check if hypergraph is 2-colorable using precomputed kill masks.
        Returns True if colorable, False if not.
        """
        if not edges:
            return True
        
        total_colorings = 1 << n
        # Bitset of colorings that are still valid (not killed by any edge yet)
        # Initially all are valid
        valid_colorings = (1 << total_colorings) - 1
        
        for e in edges:
            # Get the kill mask for this specific edge tuple
            # kill_masks_by_edge maps edge tuple to kill mask int
            km = kill_masks_by_edge.get(e)
            if km is None:
                # Should not happen if we precomputed all
                return False 
            # If this edge kills some valid colorings, remove them
            valid_colorings &= ~km
            if valid_colorings == 0:
                return False
        
        return True

    best_edges = None
    best_n = 0
    best_count = float('inf')

    # We will try n=9, 10, 11, 12
    # For each n, we do:
    # 1. Precompute kill masks for all 5-edges
    # 2. Run Randomized Greedy Set Cover multiple times
    # 3. Local Search (Deletion) on the best result from Greedy
    
    for n in [9, 10, 11, 12]:
        if time.time() > deadline - 15:
            break
            
        total_colorings = 1 << n
        
        # Generate all possible 5-edges
        all_edges = list(combinations(range(n), 5))
        
        # Precompute kill masks for each edge
        # kill_masks_by_edge: dict mapping edge tuple -> int (bitset of killed colorings)
        kill_masks_by_edge = {}
        # Also keep a list of (edge, kill_mask) for fast iteration in Greedy
        edge_kill_list = []
        
        for e in all_edges:
            e_mask = 0
            for v in e:
                e_mask |= (1 << v)
                
            km = 0
            # Optimization: Iterate colorings
            for c in range(total_colorings):
                if (c & e_mask) == 0 or (c & e_mask) == e_mask:
                    km |= (1 << c)
            kill_masks_by_edge[e] = km
            edge_kill_list.append((e, km))
            
        # Greedy Set Cover with Randomization
        # We run multiple restarts
        # Increase restarts for smaller n where computation is faster
        if n == 9:
            num_restarts = 50
        elif n == 10:
            num_restarts = 30
        elif n == 11:
            num_restarts = 15
        else:
            num_restarts = 5

        best_greedy_edges = None
        best_greedy_count = float('inf')
        
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            remaining = (1 << total_colorings) - 1
            selected_edges = []
            selected_set = set()
            
            # Shuffle edges for randomization?
            # Or just randomize tie-breaking.
            # Let's shuffle the list of candidates at each step? 
            # That's expensive.
            # Instead, we can add a small random noise to the count.
            
            while remaining:
                if time.time() > deadline - 5:
                    break
                    
                best_idx = -1
                best_count_killed = 0
                best_new_remaining = remaining
                best_edge = None
                candidates = [] # To handle ties
                
                # Iterate through all edges
                # To speed up, we can skip edges that kill 0 new colorings
                # But checking that is the cost.
                
                for i, (e, km) in enumerate(edge_kill_list):
                    if e in selected_set:
                        continue
                        
                    new_killed = remaining & km
                    if new_killed:
                        cnt = new_killed.bit_count()
                        
                        if cnt > best_count_killed:
                            best_count_killed = cnt
                            best_new_remaining = remaining & ~km
                            candidates = [(i, e, km)]
                        elif cnt == best_count_killed:
                            candidates.append((i, e, km))
                
                if best_idx == -1 and not candidates:
                    # Failure
                    break
                    
                if candidates:
                    # Pick randomly from candidates with max count
                    idx, e, km = random.choice(candidates)
                    best_edge = e
                    best_new_remaining = remaining & ~km
                else:
                    break

                selected_edges.append(best_edge)
                selected_set.add(best_edge)
                remaining = best_new_remaining
                
            if remaining == 0:
                if len(selected_edges) < best_greedy_count:
                    best_greedy_count = len(selected_edges)
                    best_greedy_edges = selected_edges[:]
                
                # Local Search: Deletion
                current_edges = best_greedy_edges[:]
                changed = True
                # Limit deletion steps to avoid timeout
                max_deletion_steps = 200
                steps = 0
                while changed and steps < max_deletion_steps and time.time() < deadline - 5:
                    changed = False
                    steps += 1
                    # Shuffle order of deletion attempts to avoid bias
                    random.shuffle(current_edges)
                    for i in range(len(current_edges)):
                        # Try removing current_edges[i]
                        test_edges = current_edges[:i] + current_edges[i+1:]
                        if not is_2_colorable_fast(n, test_edges, kill_masks_by_edge):
                            # Still non-2-colorable without this edge
                            current_edges = test_edges
                            changed = True
                            break # Restart loop with new size
                
                if len(current_edges) < best_greedy_count:
                    best_greedy_count = len(current_edges)
                    best_greedy_edges = current_edges[:]
                    
        if best_greedy_edges is not None:
            if best_greedy_count < best_count:
                best_count = best_greedy_count
                best_edges = best_greedy_edges
                best_n = n
                
        # If we found a very small one, we can stop
        if best_count <= 29:
             # Theoretical lower bound is 29. If we hit it, we can't do better.
             # But we might have time for other n? No, 29 is global min.
             # Break out of n loop if we find 29?
             # Yes, because we want to minimize edges. 29 is the min possible.
             break

    if best_edges is None:
        # Fallback: Complete graph on 9
        n = 9
        best_edges = list(combinations(range(n), 5))
        best_n = n
    else:
        # Final verification using the fast checker
        # We need kill_masks for best_n
        # If best_n was processed, we have it. But we lost the dict.
        # Let's just use the slow one for final check to be safe, or re-compute.
        # Since best_n is small, slow check is fine.
        
        def is_2_colorable_slow(n, edges):
            if not edges:
                return True
            total_colorings = 1 << n
            edge_masks = []
            for e in edges:
                mask = 0
                for v in e:
                    mask |= (1 << v)
                edge_masks.append(mask)
            for mask in range(total_colorings):
                valid = True
                for em in edge_masks:
                    if (mask & em) == 0 or (mask & em) == em:
                        valid = False
                        break
                if valid:
                    return True
            return False

        if is_2_colorable_slow(best_n, best_edges):
            # Should not happen
            n = 9
            best_edges = list(combinations(range(n), 5))
            best_n = n

    # Format output
    final_edges = [list(e) for e in best_edges]
    
    return {'n': best_n, 'edges': final_edges}

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, edges={len(res['edges'])}")

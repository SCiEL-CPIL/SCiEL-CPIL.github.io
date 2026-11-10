import math
import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # Keep 10s safety margin
    
    # 1. Generate candidates: Numbers in [1, 255] sharing a factor with 255
    # 255 = 3 * 5 * 17
    candidates = [x for x in range(1, 256) if math.gcd(x, 255) > 1]
    
    # 255 is always in the set
    if 255 not in candidates:
        candidates.append(255)
        
    n_cand = len(candidates)
    
    # Precompute gcd matrix or just use math.gcd on the fly for small N
    # N is small (~127), so on-the-fly is fine, but let's precompute neighbors for speed
    neighbors = [set() for _ in range(n_cand)]
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if math.gcd(candidates[i], candidates[j]) > 1:
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    best_set = []
    best_size = 0
    
    # Base heuristic: Greedy
    def run_greedy(initial_ordering=None):
        if initial_ordering is None:
            # Sort by degree descending
            degrees = [len(neighbors[i]) for i in range(n_cand)]
            order = sorted(range(n_cand), key=lambda x: -degrees[x])
        else:
            order = initial_ordering
            
        selected = []
        selected_set = set()
        
        # We must include 255
        idx_255 = candidates.index(255)
        selected.append(candidates[idx_255])
        selected_set.add(idx_255)
        
        # Candidates compatible with all selected
        # Initially, anyone connected to 255 (which is everyone in our candidate list)
        available = set(range(n_cand))
        available.remove(idx_255)
        
        while available:
            if time.time() > deadline:
                return selected
            
            # Find the node in available that has the most connections to 'selected_set'
            # Ties broken by total degree
            best_node = -1
            max_compat_count = -1
            max_degree = -1
            
            # Optimization: We can just pick the one with highest degree in the available set 
            # that is connected to all selected.
            # Since 255 is universal, all available are connected to 255.
            # But as we add more, the "compatible" set shrinks.
            
            # A better greedy: Pick node with highest degree in the induced subgraph of 'available'
            # that is also connected to all 'selected'.
            # Note: If it's connected to all 'selected', it's a valid addition.
            
            for node in available:
                # Check if node connects to all selected
                is_compat = True
                for sel in selected_set:
                    if sel not in neighbors[node]:
                        is_compat = False
                        break
                if not is_compat:
                    continue
                
                deg = len(neighbors[node] & available)
                # Heuristic: prefer nodes that keep the available set large?
                # Or just max degree in available?
                if deg > max_degree:
                    max_degree = deg
                    best_node = node
                    
            if best_node == -1:
                break
                
            selected.append(candidates[best_node])
            selected_set.add(best_node)
            
            # Update available: must be connected to new node
            new_available = set()
            for node in available:
                if node == best_node:
                    continue
                if best_node in neighbors[node]:
                    # And still connected to all other selected?
                    # Since 'available' was filtered to be connected to all previous selected,
                    # and 'best_node' is in selected, we just need to ensure 'node' connects to 'best_node'.
                    # Wait, 'node' must connect to ALL selected.
                    # Previous available nodes connected to all OLD selected.
                    # New selected is best_node.
                    # So we just filter available by connection to best_node.
                    new_available.add(node)
            available = new_available
            
        return selected

    # Run standard greedy
    current_best = run_greedy()
    if len(current_best) > best_size:
        best_size = len(current_best)
        best_set = current_best

    # Try randomized restarts to escape local optima
    # We can shuffle the tie-breaking in the greedy step
    for _ in range(50): # Limit restarts
        if time.time() > deadline:
            break
        
        # Randomize the order of evaluation in the greedy step
        # We do this by randomly perturbing the "degree" metric slightly
        def run_greedy_randomized():
            selected = [255]
            selected_set = {candidates.index(255)}
            available = set(range(n_cand))
            available.remove(candidates.index(255))
            
            while available:
                if time.time() > deadline:
                    return selected
                
                best_node = -1
                best_score = -1
                
                # Collect candidates compatible with all selected
                compatible = []
                for node in available:
                    if all(sel in neighbors[node] for sel in selected_set):
                        compatible.append(node)
                
                if not compatible:
                    break
                
                # Score by degree in available + random noise
                for node in compatible:
                    deg = len(neighbors[node] & available)
                    # Add small random noise to break ties and explore
                    score = deg + random.random() * 0.1
                    if score > best_score:
                        best_score = score
                        best_node = node
                        
                selected.append(candidates[best_node])
                selected_set.add(best_node)
                
                # Filter available
                available = {node for node in available if node != best_node and best_node in neighbors[node]}
                
            return selected

        current_best = run_greedy_randomized()
        if len(current_best) > best_size:
            best_size = len(current_best)
            best_set = current_best

    # Final verification (optional but good for debugging, disabled for speed in prod)
    # verify(best_set)
    
    return best_set

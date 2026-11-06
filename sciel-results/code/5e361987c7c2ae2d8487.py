import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    k4s = list(itertools.combinations(range(n), 4))
    k4_edges = []
    for k4 in k4s:
        k4_edge_indices = []
        for i in range(4):
            for j in range(i + 1, 4):
                u, v = k4[i], k4[j]
                k4_edge_indices.append(edge_index[(u, v)])
        k4_edges.append(k4_edge_indices)
        
    num_k4s = len(k4s)
    
    # Precompute edge_to_k4s
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4_edges):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)
    
    # Precompute as numpy arrays for speed
    k4_edges_np = np.array(k4_edges, dtype=np.int32)
    
    def is_valid(colours):
        """Check if colouring is valid: every K4 has >= 5 distinct colours"""
        for e_indices in k4_edges:
            cols = set(colours[e] for e in e_indices)
            if len(cols) < 5:
                return False
        return True
    
    def count_bad_k4s(colours):
        """Count number of K4s with < 5 distinct colours"""
        bad = 0
        for e_indices in k4_edges:
            cols = set(colours[e] for e in e_indices)
            if len(cols) < 5:
                bad += 1
        return bad
    
    def count_bad_k4s_fast(colours):
        """Faster version using numpy"""
        bad = 0
        for k4_idx in range(num_k4s):
            e_indices = k4_edges_np[k4_idx]
            cols = set(colours[e] for e in e_indices)
            if len(cols) < 5:
                bad += 1
        return bad
    
    # Try to find solutions for K = 13, 14, 15
    # Start with structured approaches, then local search
    
    def try_structured_13():
        """Try algebraic construction for 13 colours"""
        # Use Z_13 additive group structure
        # Map 16 vertices to elements, use cyclic differences
        # This is a heuristic - may not work directly
        
        # Alternative: use known constructions from literature
        # For now, try random + local search
        
        return None
    
    def local_search(K, max_time=30.0):
        """Local search to find valid K-colouring"""
        start = time.time()
        max_iter = 500000
        
        # Initialize
        colours = np.random.randint(0, K, size=num_edges, dtype=np.int32)
        
        # Compute initial badness
        def compute_badness(colours):
            badness = np.zeros(num_k4s, dtype=np.int32)
            for k4_idx in range(num_k4s):
                e_indices = k4_edges_np[k4_idx]
                cols = set(int(colours[e]) for e in e_indices)
                badness[k4_idx] = max(0, 5 - len(cols))
            return badness
        
        badness = compute_badness(colours)
        total_bad = int(badness.sum())
        
        if total_bad == 0:
            return colours.tolist()
        
        # Keep track of bad K4 indices
        bad_k4_indices = np.where(badness > 0)[0]
        
        # Simulated annealing
        temperature = 1.0
        cooling = 0.9999
        best_colours = colours.copy()
        best_bad = total_bad
        
        for iteration in range(max_iter):
            if time.time() - start > max_time:
                break
            
            if total_bad == 0:
                return colours.tolist()
            
            # Pick a random bad K4
            if len(bad_k4_indices) == 0:
                return colours.tolist()
            
            k4_idx = bad_k4_indices[random.randint(0, len(bad_k4_indices) - 1)]
            e_indices = k4_edges_np[k4_idx]
            
            # Pick a random edge in this K4
            e_idx = int(e_indices[random.randint(0, 5)])
            old_colour = int(colours[e_idx])
            
            # Try new colour
            new_colour = random.randint(0, K - 1)
            if new_colour == old_colour:
                continue
            
            colours[e_idx] = new_colour
            
            # Update badness for affected K4s
            delta = 0
            affected = edge_to_k4s[e_idx]
            for ak4 in affected:
                old_deficit = int(badness[ak4])
                e_inds = k4_edges_np[ak4]
                cols = set(int(colours[e]) for e in e_inds)
                new_deficit = max(0, 5 - len(cols))
                delta += new_deficit - old_deficit
                badness[ak4] = new_deficit
            
            new_total = total_bad + delta
            
            # Accept or reject
            if delta <= 0 or random.random() < np.exp(-delta / temperature):
                total_bad = new_total
                if total_bad < best_bad:
                    best_bad = total_bad
                    best_colours = colours.copy()
                # Update bad_k4_indices
                bad_k4_indices = np.where(badness > 0)[0]
            else:
                # Revert
                colours[e_idx] = old_colour
                for ak4 in affected:
                    e_inds = k4_edges_np[ak4]
                    cols = set(int(colours[e]) for e in e_inds)
                    badness[ak4] = max(0, 5 - len(cols))
            
            temperature *= cooling
        
        if best_bad == 0:
            return best_colours.tolist()
        return None
    
    # Try K=13 first with multiple restarts
    for K in [13, 14, 15]:
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 5:
            break
        
        num_restarts = min(10, int(remaining_time / 10))
        for restart in range(num_restarts):
            if time.time() - start_time > deadline - 5:
                break
            
            result = local_search(K, max_time=min(12.0, deadline - (time.time() - start_time)))
            if result is not None:
                # Verify
                if is_valid(result):
                    return result
        
        # If we get here, K=13 or 14 didn't work, try next
    
    # Fallback: return a known valid 17-colouring
    # (i+j) % 17 for edges (i,j)
    # Verify this is valid
    fallback = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    if is_valid(fallback):
        return fallback
    
    # Last resort: brute force check and return best effort
    # This should not happen
    return fallback

# Quick self-test if run directly
if __name__ == "__main__":
    result = construct()
    print(f"Number of colours used: {max(result) + 1}")
    print(f"First 20 colours: {result[:20]}")

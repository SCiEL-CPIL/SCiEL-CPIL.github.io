import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    n = 20
    vertices = list(range(n))
    
    # Precompute triples for validation
    triples = list(combinations(vertices, 3))
    num_triples = len(triples)
    
    # Convert triples to numpy array for vectorized operations if needed, 
    # but for n=20, a loop over ~1140 triples is fast enough if the inner check is optimized.
    # We will use a bitmask representation for speed in Python, but vectorized logic for the "common neighbors" count.
    
    # Represent graph as a list of integers (bitmasks)
    # adj[i] is a bitmask where bit j is 1 if (i,j) is an edge.
    
    def edge_count(adj):
        return sum(m.bit_count() for m in adj) // 2

    def is_valid(adj):
        """
        Check if the graph is K3,3-free.
        K3,3 exists if there are 3 vertices u,v,w with at least 3 common neighbors.
        """
        for u, v, w in triples:
            common = adj[u] & adj[v] & adj[w]
            if common.bit_count() >= 3:
                return False
        return True

    def can_add(u, v, adj):
        """
        Check if adding edge (u,v) maintains K3,3-free property.
        Only triples containing u or v are affected.
        """
        bit_v = 1 << v
        bit_u = 1 << u
        
        # We need to check all triples that include u or v.
        # Precomputed lists of triples per vertex would be faster, but for n=20, 
        # iterating all triples is 1140 ops. 1140 * 3 bit ops is trivial.
        # However, to be safe on speed, let's precompute vertex-to-triples mapping outside the loop.
        
        # For now, simple check.
        for u2, v2, w2 in triples:
            # If the triple doesn't contain u or v, its common neighbor count doesn't change.
            if u not in (u2, v2, w2) and v not in (u2, v2, w2):
                continue
            
            # Calculate common neighbors for this triple in the NEW graph
            # We need to simulate adding the edge.
            # It's faster to just check the count in the current graph and see if it hits 3.
            # If current count is 3, we can't add.
            # If current count is 2, we must ensure the new edge doesn't make it 3?
            # Wait. Adding an edge (u,v) increases the degree of u and v.
            # It does NOT change the set of common neighbors of any triple UNLESS u or v is in the triple.
            # If u is in the triple (u, a, b), the common neighbors are N(u) & N(a) & N(b).
            # Adding edge (u,v) adds v to N(u).
            # So the new common set is (N(u) | {v}) & N(a) & N(b) = (N(u) & N(a) & N(b)) | ({v} & N(a) & N(b)).
            # So the count increases by 1 if v is a common neighbor of a and b.
            
            a, b, c = u2, v2, w2
            
            # Get masks for a, b, c in current graph
            ma = adj[a]
            mb = adj[b]
            mc = adj[c]
            
            # Apply the hypothetical addition of edge (u,v)
            if a == u:
                ma |= bit_v
            elif a == v:
                ma |= bit_u
            if b == u:
                mb |= bit_v
            elif b == v:
                mb |= bit_u
            if c == u:
                mc |= bit_v
            elif c == v:
                mc |= bit_u
            
            common = ma & mb & mc
            if common.bit_count() >= 3:
                return False
        return True

    def mask_to_edges(adj):
        edges = []
        for i in range(n):
            m = adj[i]
            for j in range(i + 1, n):
                if m & (1 << j):
                    edges.append([i, j])
        return edges

    # Optimization: Precompute which triples contain which vertices to speed up can_add
    # Actually, for n=20, 1140 triples is small. The overhead of Python loop is the main cost.
    # Let's stick to the simple can_add but ensure it's called efficiently.
    
    # Simulated Annealing Parameters
    initial_temp = 100.0
    cooling_rate = 0.99
    min_temp = 1.0
    max_steps = 50000
    
    rng = random.Random(42)
    
    # Initialization: Greedy Random
    # Start with empty graph and add edges randomly if valid.
    adj = [0] * n
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if can_add(u, v, adj):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    best_adj = list(adj)
    best_score = edge_count(best_adj)
    current_score = best_score
    current_adj = list(adj)
    current_temp = initial_temp
    
    # Precompute neighbors list for faster swap checks? 
    # Swapping (u,v) -> (a,b) means removing u-v and adding a-b.
    # This is complex to validate incrementally. 
    # Instead, we will use "Add/Remove" moves.
    # Move 1: Add a random missing edge.
    # Move 2: Remove a random existing edge.
    # Move 3: Swap an existing edge with a missing edge (remove one, add another).
    
    # To make SA effective, we need to be able to go to lower scores (remove edges) to escape traps.
    
    steps = 0
    while current_temp > min_temp and steps < max_steps:
        if time.time() > deadline:
            break
            
        # Pick a random move
        move_type = rng.random()
        
        # Generate candidate
        cand_adj = list(current_adj)
        
        if move_type < 0.5:
            # Try to add an edge
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (current_adj[i] & (1 << j))]
            if not missing:
                current_temp *= cooling_rate
                continue
            u, v = rng.choice(missing)
            if can_add(u, v, current_adj):
                cand_adj[u] |= (1 << v)
                cand_adj[v] |= (1 << u)
            else:
                # Invalid move, skip or treat as huge penalty? 
                # In SA, we only consider valid states for K3,3 constraint? 
                # The constraint is hard. We must stay in the feasible set.
                # So if can_add is false, this move is invalid. 
                # We should probably just skip it or try another.
                # To keep the loop tight, we'll just decrement temp and continue.
                current_temp *= cooling_rate
                steps += 1
                continue
                
        elif move_type < 0.8:
            # Try to remove an edge
            existing = [(i, j) for i in range(n) for j in range(i + 1, n) if current_adj[i] & (1 << j)]
            if not existing:
                current_temp *= cooling_rate
                continue
            u, v = rng.choice(existing)
            cand_adj[u] &= ~(1 << v)
            cand_adj[v] &= ~(1 << u)
            # Removing an edge always maintains K3,3-free property (monotonicity)
            
        else:
            # Swap: Remove one, add another
            existing = [(i, j) for i in range(n) for j in range(i + 1, n) if current_adj[i] & (1 << j)]
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (current_adj[i] & (1 << j))]
            if not existing or not missing:
                current_temp *= cooling_rate
                continue
            u, v = rng.choice(existing)
            a, b = rng.choice(missing)
            
            # Remove u-v
            cand_adj[u] &= ~(1 << v)
            cand_adj[v] &= ~(1 << u)
            
            # Check if we can add a-b to the *modified* graph
            # We need to check can_add(a, b, cand_adj)
            if can_add(a, b, cand_adj):
                cand_adj[a] |= (1 << b)
                cand_adj[b] |= (1 << a)
            else:
                # Swap failed validity, revert to just removal? Or skip?
                # If we just remove, it's a valid move (lower score).
                # Let's treat it as a removal move.
                pass
        
        cand_score = edge_count(cand_adj)
        delta = cand_score - current_score
        
        # Acceptance probability
        if delta > 0:
            accept = True
        else:
            # Boltzmann distribution
            # If current_temp is very small, exp(delta/temp) is 0
            if current_temp < 1e-6:
                accept = False
            else:
                ratio = np.exp(delta / current_temp)
                accept = rng.random() < ratio
        
        if accept:
            current_adj = cand_adj
            current_score = cand_score
            
            if current_score > best_score:
                best_score = current_score
                best_adj = list(current_adj)
                
        current_temp *= cooling_rate
        steps += 1
        
    # Final polish: Greedy add on best_adj
    # We can try to add any remaining missing edges to best_adj
    for i in range(n):
        for j in range(i + 1, n):
            if not (best_adj[i] & (1 << j)):
                if can_add(i, j, best_adj):
                    best_adj[i] |= (1 << j)
                    best_adj[j] |= (1 << i)
                    
    edges = mask_to_edges(best_adj)
    return edges

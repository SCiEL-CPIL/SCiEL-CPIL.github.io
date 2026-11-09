import time
import random
import sys

def construct():
    """
    Constructs a graph with 40 vertices and 401 edges minimizing the largest book size.
    Strategy: Simulated Annealing with random restarts.
    State: Adjacency matrix (list of sets) or bitmasks.
    Objective: Maximize the number of edges while keeping max book size <= K.
    We search for the smallest K such that a graph with 401 edges exists with max book <= K.
    Based on theory, K is likely 7. We try to find a graph with max book 7 (or lower).
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    n = 40
    target_edges = 401
    
    # Helper to calculate max book size and potentially count
    def get_max_book(adj):
        # adj is a list of sets or bitmasks. Let's use bitmasks for speed in intersection.
        # But for simplicity in SA, let's use sets or just recompute.
        # Recomputing max book for 40 vertices is fast enough if we do it smartly.
        # Book size of edge (u,v) is |N(u) & N(v)|.
        max_book = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    # Count common neighbors
                    common = adj[u] & adj[v]
                    # Clear u and v from common if they are in it? 
                    # N(u) contains v, N(v) contains u. 
                    # Intersection N(u) & N(v) contains w if w is neighbor of both.
                    # It does NOT contain u or v unless self-loops (not allowed).
                    # So popcount(common) is the book size.
                    b = common.bit_count()
                    if b > max_book:
                        max_book = b
        return max_book

    def count_edges(adj):
        c = 0
        for u in range(n):
            c += adj[u].bit_count()
        return c // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    best_score = float('inf')
    best_adj = None

    while time.time() - start_time < time_limit:
        # Random Restart
        # Start with a random graph or a specific structure?
        # Let's start with a random graph with ~400 edges to be close to target.
        # Or start with K20,20 + random edges and perturb?
        # Let's try starting with a random bipartite-like structure plus noise.
        
        adj = [0] * n
        
        # Initial state: Random graph with p=0.5? No, we need exactly 401 edges.
        # Let's generate a random graph with 401 edges.
        edges = set()
        while len(edges) < target_edges:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                if u > v: u, v = v, u
                edges.add((u, v))
        
        for u, v in edges:
            add_edge(adj, u, v)
        
        current_score = get_max_book(adj)
        if current_score < best_score:
            best_score = current_score
            best_adj = [a for a in adj]
            if best_score <= 6: # Theoretical lower bound is 40/6 ~ 6.6, so integer is 7. If we find 6, it's huge.
                # Actually, Edwards/Kh... says >= 40/6. So 7 is the theoretical min integer.
                # If we find 7, we are likely optimal.
                break
        
        # Simulated Annealing
        T = 1.0
        cooling_rate = 0.999
        steps = 0
        max_steps = 50000
        
        while steps < max_steps and time.time() - start_time < time_limit:
            # Pick a random move: Add edge, Remove edge, or Swap edge
            move_type = random.randint(0, 2)
            
            if move_type == 0: # Add edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v: continue
                if adj[u] & (1 << v): continue # Already exists
                
                # Calculate delta in max book? Hard to do incrementally for MAX.
                # We just apply, check score, revert if worse (or accept with prob).
                add_edge(adj, u, v)
                new_score = get_max_book(adj)
                
                # We want to MINIMIZE max book.
                # But we also MUST maintain 401 edges.
                # So if we add, we must remove another one to keep count constant?
                # Or we can allow edge count to vary slightly and penalize?
                # Better: Do swaps.
                # Let's stick to swaps to maintain edge count strictly.
                remove_edge(adj, u, v)
                continue

            elif move_type == 1: # Remove edge
                # Pick a random existing edge
                u = random.randint(0, n - 1)
                if adj[u] == 0: continue
                v_bit = random.choice([b for b in range(n) if (adj[u] >> b) & 1])
                if u > v_bit: u, v_bit = v_bit, u
                if not (adj[u] & (1 << v_bit)): continue
                
                remove_edge(adj, u, v_bit)
                new_score = get_max_book(adj)
                
                # Now we have 400 edges. We must add one back to keep 401.
                # This is a 2-step move.
                # Let's just do Swap: Remove (u,v), Add (x,y).
                # This is complex. Let's simplify:
                # Move: Pick edge (u,v) to remove, and non-edge (x,y) to add.
                # This is a "swap".
                
                # Revert remove
                add_edge(adj, u, v_bit)
                continue

            else: # Swap edge
                # Pick random edge (u,v)
                u = random.randint(0, n - 1)
                if adj[u] == 0: continue
                # Pick random neighbor v of u
                neighbors = [b for b in range(n) if (adj[u] >> b) & 1]
                if not neighbors: continue
                v = random.choice(neighbors)
                
                # Pick random non-edge (x,y)
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y: continue
                if adj[x] & (1 << y): continue # Must be non-edge
                
                # Apply swap
                remove_edge(adj, u, v)
                add_edge(adj, x, y)
                
                new_score = get_max_book(adj)
                
                # Check edge count? Swap preserves count.
                
                delta = new_score - current_score
                
                if delta <= 0:
                    # Better or equal
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_adj = [a for a in adj]
                        if best_score <= 7:
                            # Likely optimal, break out of SA
                            break
                else:
                    # Worse, accept with probability exp(-delta/T)
                    if random.random() < pow(sys.e, -delta / T):
                        current_score = new_score
                
                T *= cooling_rate
                steps += 1
        
        # End of SA loop for this restart
        if best_score <= 7:
            break

    # Return the best graph found
    # Construct list of edges from best_adj
    result = []
    if best_adj is None:
        # Fallback to seed if something went wrong
        for a in range(20):
            for b in range(20, 40):
                result.append([a, b])
        result.append([0, 1])
    else:
        for u in range(n):
            for v in range(u + 1, n):
                if best_adj[u] & (1 << v):
                    result.append([u, v])
                    
    # Verify exactly 401 edges
    if len(result) != 401:
        # Fallback
        for a in range(20):
            for b in range(20, 40):
                result.append([a, b])
        result.append([0, 1])

    return result

# Keep the function name as required
# The prompt says "Keep the function names and signatures of the seed program."
# Seed program: def construct():

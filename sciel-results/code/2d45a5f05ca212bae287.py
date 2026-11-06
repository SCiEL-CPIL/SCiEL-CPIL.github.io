import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s kill
    
    n = 20
    total_pairs = n * (n - 1) // 2
    
    # Bitmask representation for adjacency
    # adj[i] is an integer where bit j is set if (i,j) is an edge
    def get_common_count(adj, u, v):
        """Count common neighbors of u and v using bitmasks. O(1) bitwise ops."""
        # Neighbors of u: adj[u]
        # Neighbors of v: adj[v]
        # Common: adj[u] & adj[v]
        # Popcount
        return (adj[u] & adj[v]).bit_count()
    
    def is_valid_add(adj, u, v):
        """Check if adding edge (u,v) keeps the graph K2,3-free."""
        # 1. Common neighbors of u and v themselves must be <= 2
        if get_common_count(adj, u, v) > 2:
            return False
        
        # 2. For every other w, the new common neighbor counts involving u or v must not exceed 2
        # When we add (u,v):
        # - For any w != u, v:
        #   c(u,w) increases by 1 if v is in N(w) (i.e., w in adj[v])
        #   c(v,w) increases by 1 if u is in N(w) (i.e., w in adj[u])
        
        # We can check this efficiently.
        # Let's iterate w. n=20 is small, so a loop is fine, but we can optimize.
        # Actually, just checking the specific pairs is fast enough.
        for w in range(n):
            if w == u or w == v:
                continue
            
            # Check c(u,w)
            # Current c(u,w) = (adj[u] & adj[w]).bit_count()
            # New c(u,w) = Current + (1 if v in adj[w] else 0)
            # v in adj[w] is equivalent to w in adj[v]
            inc_uw = 1 if (adj[v] >> w) & 1 else 0
            if (adj[u] & adj[w]).bit_count() + inc_uw > 2:
                return False
            
            # Check c(v,w)
            # Current c(v,w) = (adj[v] & adj[w]).bit_count()
            # New c(v,w) = Current + (1 if u in adj[w] else 0)
            # u in adj[w] is equivalent to w in adj[u]
            inc_vw = 1 if (adj[u] >> w) & 1 else 0
            if (adj[v] & adj[w]).bit_count() + inc_vw > 2:
                return False
        
        return True

    def is_valid_remove(adj, u, v):
        """Removing an edge always preserves validity if it was valid before."""
        return True

    def edge_count(adj):
        return sum(a.bit_count() for a in adj) // 2

    def to_edge_list(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Initialize
    # Start with a random dense graph and clean it quickly using the fast checker?
    # Or start with a known good sparse structure and grow?
    # Let's start with a random graph with p=0.5 and clean it using a simple greedy removal
    # that uses the fast checker to decide which edge to remove.
    
    def clean_graph(adj):
        """Remove edges to make graph valid."""
        changed = True
        while changed:
            changed = False
            # Find a violation
            violation = None
            for u in range(n):
                for v in range(u + 1, n):
                    if get_common_count(adj, u, v) > 2:
                        violation = (u, v)
                        break
                if violation:
                    break
            
            if not violation:
                break
            
            u, v = violation
            # Find a common neighbor w to break
            common = (adj[u] & adj[v])
            # Pick a random common neighbor w
            # Get list of bits set in common
            common_list = []
            w = 0
            c = common
            while c:
                if c & 1:
                    common_list.append(w)
                c >>= 1
                w += 1
            
            if not common_list:
                # Should not happen if count > 2
                break
            
            # Remove edge (u, w) or (v, w)
            w = random.choice(common_list)
            # Prefer removing edge from lower degree vertex? Or random.
            if random.random() < 0.5:
                # Remove (u, w)
                adj[u] &= ~(1 << w)
                adj[w] &= ~(1 << u)
            else:
                # Remove (v, w)
                adj[v] &= ~(1 << w)
                adj[w] &= ~(1 << v)
            changed = True
        
        return adj

    # Simulated Annealing
    best_adj = None
    best_edges = []
    
    # Run multiple restarts or one long SA?
    # Given 118s, we can do a few restarts with SA.
    
    for restart in range(5):
        if time.time() > deadline - 2:
            break
        
        # Initialize adj
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.6:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Clean
        adj = clean_graph(adj)
        
        current_energy = -edge_count(adj) # Minimize energy = Maximize edges
        best_energy = current_energy
        best_adj_restart = adj[:]
        
        # SA Parameters
        T = 1.0
        cooling = 0.995
        min_T = 0.01
        max_steps = 200000 # Adjust based on time
        
        step = 0
        while T > min_T and time.time() < deadline - 5:
            # Pick a random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            is_present = (adj[u] >> v) & 1
            
            if is_present:
                # Try removing
                # Always valid to remove
                new_energy = current_energy + 1 # Removing an edge decreases edge count, so energy increases by 1 (since energy = -count)
                # Wait, energy = -count. If we remove, count decreases by 1, so new_count = count - 1.
                # new_energy = -(count - 1) = -count + 1 = current_energy + 1.
                # Delta = new_energy - current_energy = 1.
                delta = 1
                # Accept if delta < 0 or random < exp(-delta/T)
                if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_adj_restart = adj[:]
            else:
                # Try adding
                if is_valid_add(adj, u, v):
                    new_energy = current_energy - 1 # Adding edge increases count, energy decreases by 1
                    delta = -1
                    # Always accept if valid and improves? 
                    # In SA, we accept improvements always.
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_adj_restart = adj[:]
                else:
                    # Invalid add. Can we accept it temporarily?
                    # No, the constraint is hard. We cannot have a K2,3 at any point.
                    # So we simply reject.
                    pass
            
            T *= cooling
            step += 1
        
        # Update global best
        if best_energy < (best_edges and -len(best_edges) or float('inf')):
            # best_energy is -count. So lower is better.
            # best_edges length is count.
            # Check if best_adj_restart has more edges than current best_edges
            if best_adj_restart:
                m = edge_count(best_adj_restart)
                if not best_edges or m > len(best_edges):
                    best_edges = to_edge_list(best_adj_restart)
    
    # Final fallback: if best_edges is empty (shouldn't happen), generate a trivial one
    if not best_edges:
        # Cycle C20 is K2,3 free
        for i in range(n):
            best_edges.append([i, (i + 1) % n])
    
    return best_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
    # Validate
    adj = [0] * 20
    for u, v in e:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    valid = True
    for i in range(20):
        for j in range(i + 1, 20):
            if (adj[i] & adj[j]).bit_count() > 2:
                valid = False
                print(f"Violation: {i}, {j} have {(adj[i] & adj[j]).bit_count()} common neighbors")
    if valid:
        print("Valid")
    else:
        print("Invalid")

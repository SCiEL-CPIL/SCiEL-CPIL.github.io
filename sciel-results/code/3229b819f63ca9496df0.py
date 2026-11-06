import time
import random
import sys

def construct():
    """
    Constructs a simple graph on 20 vertices with edge colours 0..19.
    Maximizes the number of edges while ensuring every 4-cycle is rainbow.
    
    Strategy:
    1. Start with a dense random graph or a C4-free graph.
    2. Use Simulated Annealing with moves:
       - Add an edge (if it doesn't create a non-rainbow 4-cycle, or if we can fix colors)
       - Remove an edge
       - Swap colors of two edges
    3. The objective is to maximize edge count while keeping the graph "valid" (all 4-cycles rainbow).
    4. We use a penalty function: Score = Edges - Penalty * (Number of Non-Rainbow 4-Cycles).
       Actually, the task requires validity. So we should only accept states that are valid, 
       or use a very high penalty to stay in the valid region.
       
       Better approach: 
       - State: (Adjacency Matrix, Color Matrix)
       - Move: 
         1. Try to add a random edge. If it creates a non-rainbow 4-cycle, reject OR try to fix by recoloring.
         2. Recolor an edge.
         3. Remove an edge.
       
       Since checking all 4-cycles is expensive (O(n^4) ~ 100k checks per move is too slow for 150s if we do many moves),
       we need an efficient incremental check.
       
       However, for n=20, the number of 4-cycles is bounded. 
       Max 4-cycles in K_20 is C(20,4) * 3 = 4845 * 3 = 14535.
       Checking 14535 4-cycles per move is feasible if optimized in C/NumPy, but in pure Python it might be tight.
       
       Let's optimize the validity check.
       A 4-cycle is non-rainbow if any two edges in the cycle share a color.
       
       Alternative Strategy: 
       The current best score is 2.21. The theoretical max edges for C4-free is ~ n^(1.5) ~ 20^1.5 = 90.
       But we don't have to be C4-free. We just need rainbow 4-cycles.
       This allows MORE edges than C4-free graphs.
       
       Let's use a Simulated Annealing approach that:
       1. Maintains a valid graph.
       2. Tries to add edges. If adding an edge creates a conflict, it tries to resolve the conflict by recoloring edges involved in the new 4-cycles.
       3. If it can't resolve, it rejects the move.
       
       To speed up, we precompute all 4-cycles.
    """
    n = 20
    deadline = time.time() + 145  # Safety margin
    
    # Precompute all 4-cycles
    # A 4-cycle is a set of 4 distinct vertices {a, b, c, d} and a specific pairing of edges?
    # No, a 4-cycle subgraph is defined by 4 vertices and 4 edges.
    # In a simple graph, a set of 4 vertices can form at most 3 distinct 4-cycles (as subgraphs).
    # Actually, it's easier to iterate over all 4-tuples (a,b,c,d) with a<b<c<d? No, order matters for the cycle.
    # Let's just store the set of edges for each 4-cycle.
    # There are C(20,4) = 4845 sets of 4 vertices.
    # For each set, there are 3 ways to form a 4-cycle (perfect matchings of the remaining 4 edges? No).
    # A 4-cycle on vertices {1,2,3,4} is (1-2-3-4-1), (1-2-4-3-1), (1-3-2-4-1).
    # These correspond to the 3 perfect matchings of the 4 vertices in the complement? No.
    # The 3 cycles are:
    # 1-2-3-4-1: Edges (1,2), (2,3), (3,4), (4,1)
    # 1-2-4-3-1: Edges (1,2), (2,4), (4,3), (3,1)
    # 1-3-2-4-1: Edges (1,3), (3,2), (2,4), (4,1)
    
    cycles = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    # Cycle 1: a-b-c-d-a
                    e1 = ((a,b), (b,c), (c,d), (d,a))
                    cycles.append(e1)
                    # Cycle 2: a-b-d-c-a
                    e2 = ((a,b), (b,d), (d,c), (c,a))
                    cycles.append(e2)
                    # Cycle 3: a-c-b-d-a
                    e3 = ((a,c), (c,b), (b,d), (d,a))
                    cycles.append(e3)
                    
    # Total cycles: 4845 * 3 = 14535
    # We need to check if all edges in a cycle exist and have distinct colors.
    
    # State: 
    # adj: 20x20 boolean matrix (numpy)
    # col: 20x20 int matrix (numpy)
    
    import numpy as np
    
    def init_state():
        # Start with a random sparse graph to ensure validity
        adj = np.zeros((n, n), dtype=bool)
        col = np.zeros((n, n), dtype=int)
        # Add some random edges
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.1:
                    c = random.randint(0, 19)
                    adj[u, v] = True
                    adj[v, u] = True
                    col[u, v] = c
                    col[v, u] = c
        return adj, col

    def is_valid(adj, col):
        """Check if all existing 4-cycles are rainbow."""
        for e in cycles:
            # Check if all edges exist
            if not (adj[e[0][0], e[0][1]] and adj[e[1][0], e[1][1]] and adj[e[2][0], e[2][1]] and adj[e[3][0], e[3][1]]):
                continue
            # Check colors
            c0 = col[e[0][0], e[0][1]]
            c1 = col[e[1][0], e[1][1]]
            c2 = col[e[2][0], e[2][1]]
            c3 = col[e[3][0], e[3][1]]
            if len({c0, c1, c2, c3}) < 4:
                return False
        return True

    def count_edges(adj):
        return int(np.sum(adj) // 2)

    def try_add_edge(adj, col, u, v):
        """
        Try to add edge (u,v) with a color that makes all new 4-cycles rainbow.
        Returns True if successful, False otherwise.
        Modifies adj and col in place if successful.
        """
        if adj[u, v]:
            return True # Already exists
        
        # Find all 4-cycles that include edge (u,v)
        # A 4-cycle containing (u,v) looks like u-v-w-x-u
        # w and x must be distinct from u,v and each other.
        # Edges: (u,v), (v,w), (w,x), (x,u)
        # We need (v,w), (w,x), (x,u) to exist.
        
        # Identify candidate cycles
        # For each w in neighbors(v):
        #   For each x in neighbors(u):
        #     if x != w and w != u and w != v and x != v and x != u (already distinct)
        #     and edge (w,x) exists:
        #       This forms a cycle u-v-w-x-u
        
        new_cycles = []
        for w in np.where(adj[v, :])[0]:
            if w == u: continue
            for x in np.where(adj[u, :])[0]:
                if x == v or x == w: continue
                if adj[w, x]:
                    # Cycle edges: (u,v), (v,w), (w,x), (x,u)
                    new_cycles.append(((u,v), (v,w), (w,x), (x,u)))
        
        if not new_cycles:
            # No new 4-cycles created, any color works
            c = random.randint(0, 19)
            adj[u, v] = True
            adj[v, u] = True
            col[u, v] = c
            col[v, u] = c
            return True
        
        # We need to find a color c for (u,v) such that for all new_cycles,
        # the set of colors is distinct.
        # This is a constraint satisfaction problem on c.
        # c must be different from the other 3 colors in each cycle.
        
        # Since there are only 20 colors, we can just try all 20.
        for c in range(20):
            valid_color = True
            for cyc in new_cycles:
                c0 = c # color of (u,v)
                c1 = col[cyc[1][0], cyc[1][1]]
                c2 = col[cyc[2][0], cyc[2][1]]
                c3 = col[cyc[3][0], cyc[3][1]]
                if len({c0, c1, c2, c3}) < 4:
                    valid_color = False
                    break
            if valid_color:
                adj[u, v] = True
                adj[v, u] = True
                col[u, v] = c
                col[v, u] = c
                return True
        
        return False

    def try_recolor(adj, col, u, v, new_color):
        """
        Try to recolor edge (u,v) to new_color.
        Checks if this breaks any existing 4-cycles.
        """
        if not adj[u, v]:
            return False
        
        old_color = col[u, v]
        if old_color == new_color:
            return True
            
        # Find all 4-cycles containing (u,v)
        # Same logic as try_add_edge
        for w in np.where(adj[v, :])[0]:
            if w == u: continue
            for x in np.where(adj[u, :])[0]:
                if x == v or x == w: continue
                if adj[w, x]:
                    # Cycle: (u,v), (v,w), (w,x), (x,u)
                    c1 = col[w, v] # (v,w)
                    c2 = col[w, x]
                    c3 = col[x, u]
                    if len({new_color, c1, c2, c3}) < 4:
                        return False
        
        col[u, v] = new_color
        col[v, u] = new_color
        return True

    # Simulated Annealing
    adj, col = init_state()
    current_edges = count_edges(adj)
    best_adj = adj.copy()
    best_col = col.copy()
    best_edges = current_edges
    
    temperature = 10.0
    cooling = 0.99
    start_time = time.time()
    
    while time.time() < deadline:
        # Generate move
        move_type = random.random()
        
        if move_type < 0.7:
            # Try to add an edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if adj[u, v]: continue
            
            # Save state
            adj_old = adj.copy()
            col_old = col.copy()
            
            if try_add_edge(adj, col, u, v):
                new_edges = count_edges(adj)
                # Always accept if valid and more edges
                # If we want to be smarter, we could accept worse states? 
                # But validity is a hard constraint. 
                # So we only accept valid states.
                # Since try_add_edge ensures validity, we accept it.
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = adj.copy()
                    best_col = col.copy()
            else:
                # Restore
                adj = adj_old
                col = col_old
                
        elif move_type < 0.9:
            # Try to recolor an edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or not adj[u, v]: continue
            
            new_color = random.randint(0, 19)
            if col[u, v] == new_color: continue
            
            # Save state
            adj_old = adj.copy()
            col_old = col.copy()
            
            if try_recolor(adj, col, u, v, new_color):
                # Recoloring doesn't change edge count, but might allow future additions
                # We accept it to escape local minima in color space
                # (Though we don't have a penalty for color diversity, just validity)
                pass
            else:
                adj = adj_old
                col = col_old
                
        else:
            # Remove an edge (rarely needed, but helps if stuck)
            if current_edges == 0: continue
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or not adj[u, v]: continue
            
            adj[u, v] = False
            adj[v, u] = False
            col[u, v] = 0
            col[v, u] = 0
            current_edges -= 1
            
        temperature *= cooling
        
    # Final check and return
    if not is_valid(best_adj, best_col):
        # Fallback to a known valid C4-free graph or star
        # Let's just return a star to be safe
        return [[0, v, 0] for v in range(1, n)]
        
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u, v]:
                edges.append([u, v, best_col[u, v]])
                
    return edges

# For testing
if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s, Edges: {len(result)}")
    
    # Quick validity check
    n = 20
    adj = [set() for _ in range(n)]
    col = {}
    for u, v, c in result:
        adj[u].add(v)
        adj[v].add(u)
        col[(u,v)] = c
        col[(v,u)] = c
        
    valid = True
    # Check all 4-cycles
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    # Check 3 cycles
                    # 1: a-b-c-d-a
                    if b in adj[a] and c in adj[b] and d in adj[c] and a in adj[d]:
                        ca = col[(a,b)]
                        cb = col[(b,c)]
                        cc = col[(c,d)]
                        cd = col[(d,a)]
                        if len({ca, cb, cc, cd}) < 4:
                            valid = False
                    # 2: a-b-d-c-a
                    if b in adj[a] and d in adj[b] and c in adj[d] and a in adj[c]:
                        ca = col[(a,b)]
                        cb = col[(b,d)]
                        cc = col[(d,c)]
                        cd = col[(c,a)]
                        if len({ca, cb, cc, cd}) < 4:
                            valid = False
                    # 3: a-c-b-d-a
                    if c in adj[a] and b in adj[c] and d in adj[b] and a in adj[d]:
                        ca = col[(a,c)]
                        cb = col[(c,b)]
                        cc = col[(b,d)]
                        cd = col[(d,a)]
                        if len({ca, cb, cc, cd}) < 4:
                            valid = False
    print(f"Valid: {valid}")


"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

class BitGraph:
    """
    Maintains a graph using bitmasks for adjacency to allow fast triangle detection and scoring.
    """
    def __init__(self, n, edges):
        self.n = n
        self.adj = [0] * n
        self.deg = [0] * n
        self.edge_set = set()
        for u, v in edges:
            self.add_edge(u, v)

    def add_edge(self, u, v):
        if u == v:
            return
        if (u, v) in self.edge_set or (v, u) in self.edge_set:
            return
        self.edge_set.add((min(u, v), max(u, v)))
        self.adj[u] |= (1 << v)
        self.adj[v] |= (1 << u)
        self.deg[u] += 1
        self.deg[v] += 1

    def remove_edge(self, u, v):
        key = (min(u, v), max(u, v))
        if key not in self.edge_set:
            return
        self.edge_set.remove(key)
        self.adj[u] &= ~(1 << v)
        self.adj[v] &= ~(1 << u)
        self.deg[u] -= 1
        self.deg[v] -= 1

    def get_score(self):
        """
        Compute the maximum triangle degree sum.
        """
        max_sum = 0
        # Iterate over unique edges
        for u, v in self.edge_set:
            # Common neighbors of u and v
            common = self.adj[u] & self.adj[v]
            # Iterate over set bits in common
            while common:
                # Get lowest set bit
                lsb = common & -common
                w = lsb.bit_length() - 1
                s = self.deg[u] + self.deg[v] + self.deg[w]
                if s > max_sum:
                    max_sum = s
                common ^= lsb
        return max_sum

    def get_worst_triangle(self):
        """
        Return the triangle (u, v, w) with the maximum degree sum, or None if no triangles.
        """
        max_sum = -1
        worst_tri = None
        for u, v in self.edge_set:
            common = self.adj[u] & self.adj[v]
            while common:
                lsb = common & -common
                w = lsb.bit_length() - 1
                s = self.deg[u] + self.deg[v] + self.deg[w]
                if s > max_sum:
                    max_sum = s
                    worst_tri = (u, v, w)
                common ^= lsb
        return worst_tri, max_sum

    def is_edge(self, u, v):
        return (min(u, v), max(u, v)) in self.edge_set

    def get_non_edges(self, exclude=None):
        """
        Generate a list of non-edges. For efficiency in search, we might just pick randomly.
        """
        pass # We will generate candidates on the fly in the search loop

def _construct_baseline():
    """K_{8,8} plus one edge inside a part."""
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append((u, v))
    edges.append((0, 1))
    return edges

def _local_search(n, initial_edges, deadline):
    """
    Local search with simulated annealing.
    """
    start_time = time.time()
    
    # Initialize state
    g = BitGraph(n, initial_edges)
    current_score = g.get_score()
    best_score = current_score
    best_edges = set(g.edge_set)
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    iterations = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        iterations += 1
        if iterations % 1000 == 0:
            # Check time
            if time.time() > deadline:
                break
        
        # Generate a move: remove one edge, add one edge
        # Strategy: Target the worst triangle to break it, or random if stuck.
        
        # 1. Identify worst triangle
        worst_tri, worst_val = g.get_worst_triangle()
        
        # 2. Choose edge to remove
        if worst_tri and random.random() < 0.7:
            # Remove an edge from the worst triangle
            u, v, w = worst_tri
            # Pick the edge incident to the highest degree vertex in the triangle
            degs = [g.deg[u], g.deg[v], g.deg[w]]
            max_deg_idx = degs.index(max(degs))
            if max_deg_idx == 0:
                rem_u, rem_v = u, v
            elif max_deg_idx == 1:
                rem_u, rem_v = v, w
            else:
                rem_u, rem_v = w, u
            # Ensure it's actually an edge (it should be)
            if not g.is_edge(rem_u, rem_v):
                # Fallback to random
                rem_u, rem_v = random.choice(list(g.edge_set))
        else:
            rem_u, rem_v = random.choice(list(g.edge_set))
        
        # 3. Choose edge to add
        # Try to add an edge that doesn't create high-sum triangles.
        # Heuristic: Pick a non-edge between low-degree vertices.
        # Random sampling is often sufficient if we have many iterations.
        # Let's pick a random non-edge.
        # Optimization: Pre-compute non-edges? No, too slow to maintain.
        # Just pick random pair and check.
        add_u, add_v = None, None
        for _ in range(20): # Try a few times to find a non-edge
            cand_u, cand_v = random.choice(all_pairs)
            if not g.is_edge(cand_u, cand_v) and (cand_u, cand_v) != (rem_u, rem_v) and (cand_u, cand_v) != (rem_v, rem_u):
                add_u, add_v = cand_u, cand_v
                break
        
        if add_u is None:
            # Graph is complete? Unlikely with 65 edges on 16 nodes (max 120)
            continue
            
        # 4. Perform Swap
        g.remove_edge(rem_u, rem_v)
        g.add_edge(add_u, add_v)
        
        new_score = g.get_score()
        delta = new_score - current_score
        
        # 5. Accept/Reject
        if delta < 0 or (T > T_min and random.random() < math.exp(-delta / T)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = set(g.edge_set)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Revert
            g.remove_edge(add_u, add_v)
            g.add_edge(rem_u, rem_v)
            no_improve_count += 1
        
        T *= cooling
        
        # Optional: Restart if stuck too long?
        # The cooling schedule handles this.

    return list(best_edges)

def construct():
    """Construct a graph on 16 vertices with 65 edges minimizing max triangle degree sum."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n = 16
    target_edges = 65
    
    # Start with baseline
    initial_edges = _construct_baseline()
    
    # Run local search
    best_edges = _local_search(n, initial_edges, deadline)
    
    # Verify edge count
    edge_set = set()
    for u, v in best_edges:
        edge_set.add((min(u, v), max(u, v)))
        
    if len(edge_set) != target_edges:
        # Fallback
        best_edges = _construct_baseline()
        edge_set = set()
        for u, v in best_edges:
            edge_set.add((min(u, v), max(u, v)))
        
    return {'n': n, 'edges': [list(e) for e in sorted(edge_set)]}

# EVOLVE-BLOCK-END

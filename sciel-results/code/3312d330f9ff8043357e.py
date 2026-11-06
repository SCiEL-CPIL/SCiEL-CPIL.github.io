"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _compute_score(n, edges):
    """Compute the maximum triangle degree sum."""
    if not edges:
        return 0
    adj = [set() for _ in range(n)]
    deg = [0] * n
    edge_set = set()
    for u, v in edges:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in edge_set:
            continue
        edge_set.add((a, b))
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    
    max_sum = 0
    for u, v in edge_set:
        if u > v:
            continue
        common = adj[u] & adj[v]
        for w in common:
            s = deg[u] + deg[v] + deg[w]
            if s > max_sum:
                max_sum = s
    return max_sum

class _GraphState:
    """Maintain graph state with incremental triangle tracking for fast score updates."""
    
    def __init__(self, n, edge_set):
        self.n = n
        self.edge_set = set()
        self.adj = [set() for _ in range(n)]
        self.deg = [0] * n
        self.triangles = []  # list of (u, v, w) where u < v < w
        for u, v in edge_set:
            self._add_edge_internal(u, v)
    
    def _add_edge_internal(self, u, v):
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in self.edge_set:
            return
        self.edge_set.add((a, b))
        self.adj[u].add(v)
        self.adj[v].add(u)
        self.deg[u] += 1
        self.deg[v] += 1
        # Find new triangles: common neighbors w where w is not u or v
        common = self.adj[u] & self.adj[v]
        for w in common:
            # Ensure canonical ordering u < v < w
            t = tuple(sorted((u, v, w)))
            # Check if triangle already exists
            if t not in self._triangle_set:
                self.triangles.append(t)
                self._triangle_set.add(t)
    
    def _remove_edge_internal(self, u, v):
        a, b = (u, v) if u < v else (v, u)
        if (a, b) not in self.edge_set:
            return
        self.edge_set.discard((a, b))
        self.adj[u].discard(v)
        self.adj[v].discard(u)
        self.deg[u] -= 1
        self.deg[v] -= 1
        # Remove triangles that include edge (u,v)
        new_triangles = []
        for t in self.triangles:
            tu, tv, tw = t
            # Check if this triangle uses the edge (u,v)
            if (tu == u and tv == v) or (tu == v and tv == u) or \
               (tu == u and tw == v) or (tu == v and tw == u) or \
               (tv == u and tw == v) or (tv == v and tw == u):
                self._triangle_set.discard(t)
            else:
                new_triangles.append(t)
        self.triangles = new_triangles
    
    def add_edge(self, u, v):
        self._add_edge_internal(u, v)
    
    def remove_edge(self, u, v):
        self._remove_edge_internal(u, v)
    
    def get_score(self):
        if not self.triangles:
            return 0
        max_sum = 0
        for u, v, w in self.triangles:
            s = self.deg[u] + self.deg[v] + self.deg[w]
            if s > max_sum:
                max_sum = s
        return max_sum
    
    def get_edges(self):
        return list(self.edge_set)
    
    def __init__(self, n, edge_set):
        self.n = n
        self.edge_set = set()
        self.adj = [set() for _ in range(n)]
        self.deg = [0] * n
        self.triangles = []
        self._triangle_set = set()
        for u, v in edge_set:
            self._add_edge_internal(u, v)

def _construct_baseline():
    """K_{8,8} plus one edge inside a part."""
    edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    return edges

def _local_search(state, all_possible, max_iter, deadline, tabu_len=20):
    """Local search with tabu list."""
    current_score = state.get_score()
    best_score = current_score
    best_edges = set(state.edge_set)
    tabu = []
    tabu_set = set()
    
    existing_edges = list(state.edge_set)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try several swaps per iteration
        for _ in range(10):
            if time.time() > deadline:
                break
            
            # Pick random edge to remove
            if not existing_edges:
                break
            u1, v1 = random.choice(existing_edges)
            
            # Pick random edge to add
            u2, v2 = random.choice(all_possible)
            
            if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                continue
            if (u2, v2) in state.edge_set or (v2, u2) in state.edge_set:
                continue
            
            # Tabu check
            swap_key = (min(u1,v1), max(u1,v1), min(u2,v2), max(u2,v2))
            if swap_key in tabu_set:
                continue
            
            # Perform swap
            state.remove_edge(u1, v1)
            state.add_edge(u2, v2)
            
            new_score = state.get_score()
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_edges = set(state.edge_set)
                # Update tabu
                tabu.append(swap_key)
                tabu_set.add(swap_key)
                if len(tabu) > tabu_len:
                    old_key = tabu.pop(0)
                    tabu_set.discard(old_key)
                # Update existing edges list
                existing_edges = list(state.edge_set)
            else:
                # Revert
                state.remove_edge(u2, v2)
                state.add_edge(u1, v1)
    
    return best_score, best_edges

def _simulated_annealing(state, all_possible, max_iter, deadline):
    """Simulated annealing."""
    current_score = state.get_score()
    best_score = current_score
    best_edges = set(state.edge_set)
    
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    existing_edges = list(state.edge_set)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        if not existing_edges:
            break
            
        u1, v1 = random.choice(existing_edges)
        u2, v2 = random.choice(all_possible)
        
        if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
            continue
        if (u2, v2) in state.edge_set or (v2, u2) in state.edge_set:
            continue
        
        # Perform swap
        state.remove_edge(u1, v1)
        state.add_edge(u2, v2)
        
        new_score = state.get_score()
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = set(state.edge_set)
            existing_edges = list(state.edge_set)
        else:
            # Revert
            state.remove_edge(u2, v2)
            state.add_edge(u1, v1)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_score, best_edges

def _perturb_and_restart(n, target_edges, all_possible, deadline, max_restarts=5):
    """Multiple restarts with perturbation."""
    best_score = float('inf')
    best_edges = None
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Start from baseline
        base_edges = _construct_baseline()
        edges_set = set()
        for u, v in base_edges:
            a, b = (u, v) if u < v else (v, u)
            edges_set.add((a, b))
        
        # Apply random perturbations
        num_perturb = random.randint(3, 8)
        for _ in range(num_perturb):
            if len(edges_set) != target_edges:
                break
            # Remove a random edge
            if not edges_set:
                break
            u1, v1 = random.choice(list(edges_set))
            edges_set.discard((u1, v1))
            
            # Add a random non-existent edge
            candidates = [e for e in all_possible if e not in edges_set]
            if not candidates:
                break
            u2, v2 = random.choice(candidates)
            edges_set.add((u2, v2))
        
        if len(edges_set) != target_edges:
            continue
        
        # Run local search from this perturbed state
        state = _GraphState(n, edges_set)
        remaining_time = deadline - time.time()
        if remaining_time < 1:
            break
        
        ls_iter = max(100, int(remaining_time * 200))
        score, edges = _local_search(state, all_possible, ls_iter, deadline, tabu_len=30)
        
        if score < best_score:
            best_score = score
            best_edges = edges
    
    return best_score, best_edges

def construct():
    """Construct a graph on 16 vertices with 65 edges minimizing max triangle degree sum."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    n = 16
    target_edges = 65
    
    # Precompute all possible edges
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Start with baseline
    best_edges = _construct_baseline()
    best_score = _compute_score(n, best_edges)
    
    # Run local search from baseline
    if time.time() < deadline:
        edges_set = set()
        for u, v in best_edges:
            a, b = (u, v) if u < v else (v, u)
            edges_set.add((a, b))
        state = _GraphState(n, edges_set)
        remaining = deadline - time.time()
        ls_iter = max(500, int(remaining * 100))
        score, edges = _local_search(state, all_possible, ls_iter, deadline, tabu_len=30)
        if score < best_score:
            best_score = score
            best_edges = edges
    
    # Run simulated annealing from current best
    if time.time() < deadline:
        edges_set = set()
        for u, v in best_edges:
            a, b = (u, v) if u < v else (v, u)
            edges_set.add((a, b))
        state = _GraphState(n, edges_set)
        remaining = deadline - time.time()
        sa_iter = max(1000, int(remaining * 50))
        score, edges = _simulated_annealing(state, all_possible, sa_iter, deadline)
        if score < best_score:
            best_score = score
            best_edges = edges
    
    # Run multiple restarts with perturbation
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 10:
            num_restarts = min(5, max(1, int(remaining / 15)))
            score, edges = _perturb_and_restart(n, target_edges, all_possible, deadline, max_restarts=num_restarts)
            if score < best_score:
                best_score = score
                best_edges = edges
    
    # Final verification: ensure exactly 65 edges
    edge_set = set()
    for u, v in best_edges:
        a, b = (u, v) if u < v else (v, u)
        edge_set.add((a, b))
    
    if len(edge_set) != target_edges:
        # Fall back to baseline
        best_edges = _construct_baseline()
        edge_set = set()
        for u, v in best_edges:
            a, b = (u, v) if u < v else (v, u)
            edge_set.add((a, b))
    
    return {'n': n, 'edges': [list(e) for e in sorted(edge_set)]}
# EVOLVE-BLOCK-END

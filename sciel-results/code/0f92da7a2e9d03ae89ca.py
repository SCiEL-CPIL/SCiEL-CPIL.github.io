import time
import random
import sys

N = 128
DEADLINE_MARGIN = 5.0

def _build_edges():
    edges = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 > N or s2 > N or s3 > N:
                    continue
                edge = frozenset([a, b, c, s1, s2, s3])
                edges.append(edge)
    return edges

def construct():
    t_start = time.time()
    deadline = t_start + 60.0 - DEADLINE_MARGIN

    edges = _build_edges()
    num_edges = len(edges)
    
    vert_to_edges = {v: [] for v in range(1, N + 1)}
    for i, e in enumerate(edges):
        for v in e:
            vert_to_edges[v].append(i)
            
    # Precompute edge sizes for quick access
    edge_sizes = [len(e) for e in edges]

    def _can_add(v, edge_counts):
        for ei in vert_to_edges[v]:
            if edge_counts[ei] == edge_sizes[ei] - 1:
                return False
        return True

    def _update_counts(v, delta, edge_counts):
        for ei in vert_to_edges[v]:
            edge_counts[ei] += delta

    best_set = set()
    best_size = 0
    
    all_vertices = list(range(1, N + 1))
    
    def _greedy_add_all(cur_set, cur_edge_counts):
        """Add all possible missing vertices greedily. Returns count added."""
        added = 0
        missing = [v for v in all_vertices if v not in cur_set]
        random.shuffle(missing)
        for v in missing:
            if _can_add(v, cur_edge_counts):
                cur_set.add(v)
                _update_counts(v, 1, cur_edge_counts)
                added += 1
        return added

    def _local_search(cur_set, cur_edge_counts, cur_size):
        """Perform local search with multi-swap. Returns updated set, counts, size."""
        stuck_count = 0
        max_stuck = 30
        while time.time() < deadline:
            improved = False
            
            # 1. Try to add all possible missing vertices
            added = _greedy_add_all(cur_set, cur_edge_counts)
            if added > 0:
                cur_size += added
                stuck_count = 0
                continue
                
            # 2. No adds possible. Try multi-pair swaps.
            found_swap = False
            if cur_set:
                cur_list = list(cur_set)
                random.shuffle(cur_list)
                
                for num_rem in [1, 2, 3]:
                    if num_rem > len(cur_list):
                        continue
                    
                    rem_vertices = cur_list[:num_rem]
                    
                    old_set = cur_set.copy()
                    old_edge_counts = cur_edge_counts.copy()
                    old_size = cur_size
                    
                    for v in rem_vertices:
                        cur_set.remove(v)
                        _update_counts(v, -1, cur_edge_counts)
                    
                    missing_after = [v for v in all_vertices if v not in cur_set]
                    random.shuffle(missing_after)
                    
                    local_added = 0
                    for v in missing_after:
                        if _can_add(v, cur_edge_counts):
                            cur_set.add(v)
                            _update_counts(v, 1, cur_edge_counts)
                            local_added += 1
                    
                    new_size = cur_size - num_rem + local_added
                    
                    if new_size > old_size:
                        cur_size = new_size
                        improved = True
                        found_swap = True
                        break
                    else:
                        cur_set = old_set
                        cur_edge_counts = old_edge_counts
                        cur_size = old_size
                    
                    if found_swap:
                        break
            
            if improved:
                stuck_count = 0
            else:
                stuck_count += 1
                if stuck_count > max_stuck:
                    break
        
        return cur_set, cur_edge_counts, cur_size

    def _polish(best_set):
        """Final exhaustive polish on best_set."""
        if time.time() >= deadline:
            return best_set
        
        be = [0] * num_edges
        for v in best_set:
            _update_counts(v, 1, be)
        
        for _ in range(5):
            missing_final = [v for v in all_vertices if v not in best_set]
            random.shuffle(missing_final)
            for v in missing_final:
                if _can_add(v, be):
                    best_set.add(v)
                    _update_counts(v, 1, be)
        
        improved_final = True
        while improved_final and time.time() < deadline:
            improved_final = False
            be = [0] * num_edges
            for v in best_set:
                _update_counts(v, 1, be)
            
            for rem_v in list(best_set):
                old_set = best_set.copy()
                old_be = be.copy()
                
                best_set.remove(rem_v)
                _update_counts(rem_v, -1, be)
                
                missing_after = [v for v in all_vertices if v not in best_set]
                random.shuffle(missing_after)
                
                added = 0
                for v in missing_after:
                    if _can_add(v, be):
                        best_set.add(v)
                        _update_counts(v, 1, be)
                        added += 1
                
                if len(best_set) > len(old_set):
                    improved_final = True
                    break
                else:
                    best_set = old_set
                    be = old_be
        
        if time.time() < deadline:
            be = [0] * num_edges
            for v in best_set:
                _update_counts(v, 1, be)
            missing_final = [v for v in all_vertices if v not in best_set]
            random.shuffle(missing_final)
            for v in missing_final:
                if _can_add(v, be):
                    best_set.add(v)
                    _update_counts(v, 1, be)
        
        return best_set

    # Phase 1: Simulated Annealing (Max-SAT style)
    # Variables: x[v] = 1 if v in set, 0 otherwise
    # Clauses: For each edge (forbidden pattern), NOT(all 6 vertices selected)
    # Energy = sum over edges of (product of x[v] for v in edge)
    # We want to minimize energy (should be 0 for valid sets) while maximizing |S|
    
    def _compute_energy(x, edge_counts):
        """Compute total clause violations (energy)."""
        energy = 0
        for ei in range(num_edges):
            if edge_counts[ei] == edge_sizes[ei]:
                energy += 1
        return energy

    def _sa_search():
        """Simulated annealing to find large valid sets."""
        nonlocal best_set, best_size
        
        # Initialize: random set with bias toward larger numbers
        x = {}
        for v in range(1, N + 1):
            # Bias: higher numbers more likely to be included
            p = 0.7 if v > 64 else 0.4
            x[v] = 1 if random.random() < p else 0
        
        cur_set = set(v for v in range(1, N + 1) if x[v])
        cur_edge_counts = [0] * num_edges
        for v in cur_set:
            _update_counts(v, 1, cur_edge_counts)
        
        cur_energy = _compute_energy(x, cur_edge_counts)
        cur_size = len(cur_set)
        
        # Temperature schedule
        T_start = 10.0
        T_end = 0.01
        T_decay = 0.995
        T = T_start
        
        # Track best
        sa_best_set = cur_set.copy()
        sa_best_size = cur_size
        sa_best_energy = cur_energy
        
        iterations = 0
        max_iters = 500000
        
        while T > T_end and iterations < max_iters:
            if time.time() > deadline:
                break
            
            # Pick a random vertex to flip
            v = random.randint(1, N)
            
            # Compute delta energy if we flip v
            if x[v] == 1:
                # Removing v: energy decreases for edges that become incomplete
                delta_energy = 0
                for ei in vert_to_edges[v]:
                    if cur_edge_counts[ei] == edge_sizes[ei]:
                        # This edge was fully selected (violating), removing v fixes it
                        delta_energy -= 1
                delta_size = -1
            else:
                # Adding v: energy increases for edges that become fully selected
                delta_energy = 0
                for ei in vert_to_edges[v]:
                    if cur_edge_counts[ei] == edge_sizes[ei] - 1:
                        # This edge will become fully selected (new violation)
                        delta_energy += 1
                delta_size = 1
            
            # Objective: minimize energy, maximize size
            # Combined score: -size + lambda * energy
            lambda_energy = 10.0
            delta_score = -delta_size + lambda_energy * delta_energy
            
            # Acceptance: always accept if better, else with probability exp(-delta/T)
            if delta_score <= 0:
                accept = True
            else:
                accept = random.random() < (1.0 / (1.0 + delta_score * T))
            
            if accept:
                # Flip v
                x[v] = 1 - x[v]
                if x[v] == 1:
                    cur_set.add(v)
                    _update_counts(v, 1, cur_edge_counts)
                else:
                    cur_set.remove(v)
                    _update_counts(v, -1, cur_edge_counts)
                
                cur_energy += delta_energy
                cur_size += delta_size
                
                if cur_energy == 0 and cur_size > sa_best_size:
                    sa_best_size = cur_size
                    sa_best_set = cur_set.copy()
                    sa_best_energy = 0
            
            T *= T_decay
            iterations += 1
        
        # Update global best if SA found something better
        if sa_best_energy == 0 and sa_best_size > best_size:
            best_size = sa_best_size
            best_set = sa_best_set.copy()
        
        # Also try to improve sa_best_set with local search
        if sa_best_energy == 0 and sa_best_size > 0:
            if time.time() < deadline:
                ls_set, ls_counts, ls_size = _local_search(
                    sa_best_set.copy(), 
                    [0] * num_edges,
                    sa_best_size
                )
                # Rebuild counts for ls_set
                ls_counts = [0] * num_edges
                for v in ls_set:
                    _update_counts(v, 1, ls_counts)
                # Verify validity
                valid = True
                for ei in range(num_edges):
                    if ls_counts[ei] == edge_sizes[ei]:
                        valid = False
                        break
                if valid and ls_size > best_size:
                    best_size = ls_size
                    best_set = ls_set.copy()
        
        return sa_best_set, sa_best_size

    # Phase 2: Greedy restarts with local search
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initial set
        if r % 2 == 0:
            cur_set = set(range(65, N + 1))
        else:
            cur_set = set()
            for v in range(1, N + 1):
                if random.random() < 0.5:
                    cur_set.add(v)
        
        # Perturb
        num_remove = random.randint(0, 10)
        for _ in range(num_remove):
            if cur_set:
                v = random.choice(list(cur_set))
                cur_set.remove(v)
        
        cur_edge_counts = [0] * num_edges
        for v in cur_set:
            _update_counts(v, 1, cur_edge_counts)
            
        cur_size = len(cur_set)
        
        if cur_size > best_size:
            best_size = cur_size
            best_set = cur_set.copy()
            
        # Local search
        cur_set, cur_edge_counts, cur_size = _local_search(cur_set, cur_edge_counts, cur_size)
        
        if cur_size > best_size:
            best_size = cur_size
            best_set = cur_set.copy()
    
    # Phase 3: More SA runs if time permits
    if time.time() < deadline:
        for _ in range(3):
            if time.time() > deadline:
                break
            _sa_search()
    
    # Final polish
    if best_set:
        best_set = _polish(best_set)
    
    result = [int(v) for v in sorted(best_set)]
    return result

if __name__ == "__main__":
    print("Starting...")
    res = construct()
    print(f"Size: {len(res)}")

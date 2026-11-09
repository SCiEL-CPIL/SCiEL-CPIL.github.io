"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    deadline = 140  # safety margin before 150s kill
    
    # Build edge index mapping: (i,j) -> linear index 0..119
    N = 16
    E = N * (N - 1) // 2  # 120 edges
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = idx
            idx_to_edge.append((i, j))
            idx += 1
    
    # Precompute for each edge, which K4s it belongs to, and for each K4, its 6 edge indices
    # A K4 is determined by 4 vertices {a,b,c,d} with a<b<c<d
    # The 6 edges in K4: (a,b),(a,c),(a,d),(b,c),(b,d),(c,d)
    
    k4_list = []  # list of (frozenset of 4 vertices, list of 6 edge indices)
    for quad in combinations(range(N), 4):
        edges_in_k4 = []
        for pair in combinations(quad, 2):
            edges_in_k4.append(edge_to_idx[pair])
        k4_list.append((quad, edges_in_k4))
    
    # Precompute for each edge index, the list of K4 indices that contain it
    edge_to_k4s = [[] for _ in range(E)]
    for k4_idx, (quad, edges_in_k4) in enumerate(k4_list):
        for e in edges_in_k4:
            edge_to_k4s[e].append(k4_idx)
    
    # Solver class
    class Solver:
        def __init__(self, colors):
            self.colors = colors[:]  # list of 120 ints
            self.max_color = max(colors) if colors else 0
            # For each K4, count distinct colours among its 6 edges
            self.k4_distinct = [0] * len(k4_list)
            self.viol_count = 0
            self._recompute_all()
        
        def _recompute_all(self):
            self.viol_count = 0
            self.max_color = 0
            for e in range(E):
                c = self.colors[e]
                if c > self.max_color:
                    self.max_color = c
            for k4_idx, (quad, edges_in_k4) in enumerate(k4_list):
                col_set = set()
                for e in edges_in_k4:
                    col_set.add(self.colors[e])
                d = len(col_set)
                self.k4_distinct[k4_idx] = d
                if d < 5:
                    self.viol_count += 1
        
        def get_score(self):
            # Score: minimize max colour, but must be valid
            # Primary: viol_count (must be 0)
            # Secondary: max_color
            return (self.viol_count, self.max_color)
        
        def change_edge(self, e_idx, new_c):
            old_c = self.colors[e_idx]
            if old_c == new_c:
                return
            self.colors[e_idx] = new_c
            
            # Update max_color (approximate: just recompute if new_c > max, or if old_c was max)
            if new_c > self.max_color:
                self.max_color = new_c
            # If old_c was max_color, we might need to recompute, but for simplicity in search,
            # we'll track it loosely and do exact check at the end
            
            # Update K4 distinct counts
            for k4_idx in edge_to_k4s[e_idx]:
                edges_in_k4 = k4_list[k4_idx][1]
                # Recompute distinct count for this K4
                col_set = set()
                for e in edges_in_k4:
                    col_set.add(self.colors[e])
                old_d = self.k4_distinct[k4_idx]
                new_d = len(col_set)
                self.k4_distinct[k4_idx] = new_d
                if old_d < 5 and new_d >= 5:
                    self.viol_count -= 1
                elif old_d >= 5 and new_d < 5:
                    self.viol_count += 1
    
    def seed_colors():
        return [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]
    
    # Start with seed
    best_solver = Solver(seed_colors())
    best_score = best_solver.get_score()
    best_colors = best_solver.colors[:]
    
    if best_score[0] == 0:
        # Already valid with 17 colours. Try to reduce max colour.
        pass
    
    # Local search: try to reduce violations first, then reduce max colour
    # Use simulated annealing-like approach with random perturbations
    
    def local_search(solver, max_iters, time_limit):
        t_start = time.time()
        no_improve = 0
        current_score = solver.get_score()
        
        for it in range(max_iters):
            if time.time() - t_start > time_limit:
                break
            if time.time() > deadline:
                break
            
            # Pick a random edge
            e_idx = random.randint(0, E - 1)
            
            # Choose new colour
            if solver.viol_count > 0:
                # Try to fix violations: pick a random colour 0..15
                new_c = random.randint(0, 15)
            else:
                # Valid: try to reduce max colour
                # Pick from 0 to max_color-1 to try to lower it
                if solver.max_color > 0:
                    new_c = random.randint(0, solver.max_color - 1)
                else:
                    new_c = 0
            
            old_c = solver.colors[e_idx]
            if old_c == new_c:
                continue
            
            solver.change_edge(e_idx, new_c)
            new_score = solver.get_score()
            
            # Accept if better
            if new_score < current_score:
                current_score = new_score
                no_improve = 0
            else:
                # Revert
                solver.change_edge(e_idx, old_c)
                no_improve += 1
                
                if no_improve > 200:
                    # Perturbation: make several random changes
                    for _ in range(8):
                        pe = random.randint(0, E - 1)
                        pc = random.randint(0, 15)
                        solver.change_edge(pe, pc)
                    no_improve = 0
                    current_score = solver.get_score()
        
        return solver, current_score
    
    # Phase 1: Try to get a valid solution (viol_count = 0)
    # The seed is already valid with 17 colours. Let's verify.
    print(f"Seed: viol={best_score[0]}, max_color={best_score[1]}", flush=True)
    
    if best_score[0] > 0:
        # Need to fix violations
        solver = Solver(seed_colors())
        max_phase1 = 50000
        time_per_phase = 30
        solver, cur_score = local_search(solver, max_phase1, time_per_phase)
        if cur_score < best_score:
            best_score = cur_score
            best_colors = solver.colors[:]
        print(f"After Phase 1: viol={best_score[0]}, max_color={best_score[1]}", flush=True)
    
    # Phase 2: Reduce max colour while maintaining validity
    if best_score[0] == 0:
        solver = Solver(best_colors)
        # Multiple restarts of local search to reduce max colour
        restarts = 10
        for r in range(restarts):
            if time.time() > deadline:
                break
            solver_copy = Solver(solver.colors[:])
            time_remaining = deadline - time.time()
            time_per_restart = max(2, time_remaining / (restarts - r))
            solver_copy, cur_score = local_search(solver_copy, 100000, time_per_restart)
            if cur_score < best_score:
                best_score = cur_score
                best_colors = solver_copy.colors[:]
                print(f"Restart {r}: viol={best_score[0]}, max_color={best_score[1]}", flush=True)
            if best_score[1] <= 13:
                break
    
    # Final: recompute exact score
    final_solver = Solver(best_colors)
    final_solver._recompute_all()
    final_score = final_solver.get_score()
    
    if final_score[0] == 0:
        return list(best_colors)
    else:
        # Fallback to seed
        return seed_colors()

# EVOLVE-BLOCK-END

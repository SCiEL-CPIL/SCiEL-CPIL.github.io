"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 105.0  # safety margin before 120s kill
    d = 7
    n = 1 << d  # 128 vertices
    
    # Precompute all edges
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v))
    
    m = len(all_edges)  # 448 edges
    
    # Precompute face (4-cycle) structure
    # Each face is identified by (vertex, i, j) where i<j are the two differing dimensions
    # A face has 4 edges. We precompute:
    # face_list: list of tuples of 4 edge indices
    # edge_to_faces: for each edge, list of face indices it belongs to
    
    edge_to_idx = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_to_idx[(u, v)] = idx
    
    face_list = []
    edge_to_faces = [[] for _ in range(m)]
    
    for u in range(n):
        for i in range(d):
            for j in range(i+1, d):
                # Face with corners u, u^i, u^j, u^i^j
                a = u
                b = u ^ (1 << i)
                c = u ^ (1 << j)
                e = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(a, b), max(a, b))
                e2 = (min(a, c), max(a, c))
                e3 = (min(b, e), max(b, e))
                e4 = (min(c, e), max(c, e))
                
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                fid = len(face_list)
                face_list.append((idx1, idx2, idx3, idx4))
                edge_to_faces[idx1].append(fid)
                edge_to_faces[idx2].append(fid)
                edge_to_faces[idx3].append(fid)
                edge_to_faces[idx4].append(fid)
    
    num_faces = len(face_list)  # C(7,2) * 2^(7-2) = 21 * 32 = 672
    
    # Precompute for each edge, the 3 partner edges in each face it's in
    # edge_face_partners[edge_idx] = list of (face_idx, partner1, partner2, partner3)
    edge_face_partners = [[] for _ in range(m)]
    for fid, (e1, e2, e3, e4) in enumerate(face_list):
        edge_face_partners[e1].append((fid, e2, e3, e4))
        edge_face_partners[e2].append((fid, e1, e3, e4))
        edge_face_partners[e3].append((fid, e1, e2, e4))
        edge_face_partners[e4].append((fid, e1, e2, e3))
    
    def can_add_edge(chosen, idx):
        """Check if adding edge idx violates any face constraint."""
        for (fid, p1, p2, p3) in edge_face_partners[idx]:
            if chosen[p1] and chosen[p2] and chosen[p3]:
                return False
        return True
    
    def greedy_init(seed):
        """Randomized greedy initialization."""
        rng = random.Random(seed)
        edge_order = list(range(m))
        rng.shuffle(edge_order)
        
        chosen = np.zeros(m, dtype=bool)
        count = 0
        
        for idx in edge_order:
            if can_add_edge(chosen, idx):
                chosen[idx] = True
                count += 1
        
        return chosen, count
    
    def local_search_maxsat(chosen, count, seed):
        """
        Max-SAT inspired local search with weighted clause penalties.
        Objective: maximize sum(x_i) - lambda * sum(violated_faces)
        Use variable flipping with acceptance based on objective change.
        """
        rng = random.Random(seed)
        best_chosen = chosen.copy()
        best_count = count
        best_obj = count
        
        # Face violation count for fast updates
        face_violations = np.zeros(num_faces, dtype=np.int8)
        for fid, (e1, e2, e3, e4) in enumerate(face_list):
            if chosen[e1] and chosen[e2] and chosen[e3] and chosen[e4]:
                face_violations[fid] = 1
        
        total_violations = int(face_violations.sum())
        # Objective: count - lambda * violations
        lam = 2.0  # penalty weight
        obj = count - lam * total_violations
        
        # Simulated-annealing-like local search
        T = 1.5  # initial "temperature"
        cool = 0.9995
        
        max_iter = 50000
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random edge to flip
            idx = rng.randint(0, m-1)
            
            if chosen[idx]:
                # Try removing edge idx
                # Gain in count: -1
                # Change in violations: faces where this edge is the 4th completing edge
                # i.e., faces where the other 3 are all chosen
                viol_change = 0
                for (fid, p1, p2, p3) in edge_face_partners[idx]:
                    if chosen[p1] and chosen[p2] and chosen[p3]:
                        # This face was violated (all 4 chosen), removing idx fixes it
                        viol_change -= 1  # violation decreases
                delta_obj = -1 - lam * viol_change
            else:
                # Try adding edge idx
                # Gain in count: +1
                # Change in violations: faces where the other 3 are all chosen
                viol_change = 0
                for (fid, p1, p2, p3) in edge_face_partners[idx]:
                    if chosen[p1] and chosen[p2] and chosen[p3]:
                        viol_change += 1  # violation increases
                delta_obj = 1 - lam * viol_change
            
            # Accept if improvement, or with probability exp(delta_obj/T) if worsening
            if delta_obj >= 0:
                accept = True
            else:
                # delta_obj is negative, accept with prob exp(delta_obj / T)
                if T > 0.01:
                    accept = rng.random() < np.exp(delta_obj / T)
                else:
                    accept = False
            
            if accept:
                chosen[idx] = not chosen[idx]
                if chosen[idx]:
                    count += 1
                else:
                    count -= 1
                total_violations += -viol_change if not chosen[idx] else viol_change
                # Recompute violations properly
                # Actually let's just track it:
                # If we added: violations increased by viol_change (which was >= 0)
                # If we removed: violations decreased by |viol_change|
                # The above is getting confusing, let me just recompute the affected faces
                
                obj = count - lam * total_violations
                if count > best_count:
                    best_count = count
                    best_chosen = chosen.copy()
            
            # Cool down
            T *= cool
        
        # Final cleanup: try to add any edges that can be added without violation
        for idx in range(m):
            if not chosen[idx] and can_add_edge(chosen, idx):
                chosen[idx] = True
                count += 1
        if count > best_count:
            best_count = count
            best_chosen = chosen.copy()
        
        # Try 2-opt style: remove one edge, try to add multiple
        for idx in range(m):
            if time.time() > deadline - 5.0:
                break
            if not chosen[idx]:
                continue
            chosen[idx] = False
            count -= 1
            added = []
            for jdx in range(m):
                if chosen[jdx]:
                    continue
                if can_add_edge(chosen, jdx):
                    chosen[jdx] = True
                    count += 1
                    added.append(jdx)
            if count > best_count:
                best_count = count
                best_chosen = chosen.copy()
            else:
                for jdx in added:
                    chosen[jdx] = False
                    count -= 1
                chosen[idx] = True
                count += 1
        
        return best_chosen, best_count
    
    # Main search loop
    best_count = 0
    best_chosen = None
    
    # Phase 1: Multiple greedy inits + local search
    num_restarts = 0
    for seed in range(5000):
        if time.time() > deadline - 15.0:
            break
        chosen, count = greedy_init(seed)
        if count > best_count:
            best_count = count
            best_chosen = chosen.copy()
        num_restarts += 1
    
    # Phase 2: Local search on best and top candidates
    if best_chosen is not None and time.time() < deadline - 10.0:
        # Run local search on the best solution
        best_chosen, best_count = local_search_maxsat(best_chosen, best_count, seed=42)
    
    # Phase 3: More local search iterations with different seeds
    if time.time() < deadline - 10.0:
        for seed in [100, 200, 300, 400, 500]:
            if time.time() > deadline - 10.0:
                break
            chosen, count = greedy_init(seed + 10000)
            if count >= best_count - 5:
                chosen, count = local_search_maxsat(chosen, count, seed=seed)
                if count > best_count:
                    best_count = count
                    best_chosen = chosen.copy()
    
    # Phase 4: Final polish - try adding any remaining valid edges
    if best_chosen is not None and time.time() < deadline - 5.0:
        for idx in range(m):
            if not best_chosen[idx] and can_add_edge(best_chosen, idx):
                best_chosen[idx] = True
                best_count += 1
    
    # Convert back to list of edges
    out = []
    for idx in range(m):
        if best_chosen[idx]:
            out.append(list(all_edges[idx]))
    
    return out
# EVOLVE-BLOCK-END

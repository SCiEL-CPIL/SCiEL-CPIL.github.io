"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def generate_candidates(n, k, E_CI, limit=1000):
        """
        Generate candidate cliques for the MIP.
        A candidate is a subset S of C such that S is a clique in G[C].
        The actual clique in the partition is S ∪ {v} for some v in I where S ⊆ N(v).
        
        We categorize candidates:
        1. C-Clique candidates: S ⊆ C, S is a clique. 
           These can be used to cover C-edges.
           If we use S for vertex v, it covers C-edges in S and CI-edges (u,v) for u in S.
           
        To keep the MIP tractable, we only include:
        - All edges (size 2 cliques in C)
        - All maximal cliques of size 3 and 4 (and maybe 5 for small k)
        
        Actually, since G[C] is a complete graph, ANY subset of C is a clique.
        So the "clique constraint" on C-vertices is automatically satisfied.
        The only constraint is that if we form a clique K = S ∪ {v}, 
        then S must be a subset of N(v) (the neighbors of v in C).
        
        Wait, the graph is a split graph. C is a clique. I is an independent set.
        Edges are within C (complete) and between C and I (random).
        A clique in the graph can contain:
        - At most one vertex from I (since I is independent).
        - Any number of vertices from C (since C is a clique), BUT they must be adjacent to the I-vertex if one is present.
        
        So a clique is either:
        1. A subset of C (size >= 2). Let's call these C-cliques.
        2. A subset S of C (size >= 1) plus one vertex v in I, such that S ⊆ N(v). Let's call these I-cliques.
        
        The MIP will decide which edges are covered by which clique.
        Variables:
        x_{c, e} = 1 if edge e is covered by candidate clique c.
        
        Constraints:
        1. For each edge e, sum_{c: e in c} x_{c, e} = 1.
        2. For each candidate clique c, we need to ensure that if any edge e in c is assigned to c, then ALL edges in c are assigned to c?
           No, that's not right. A clique partition means we select a set of cliques.
           Let y_c = 1 if candidate clique c is used in the partition.
           Then x_{c, e} <= y_c.
           And sum_{c: e in c} x_{c, e} = 1.
           Minimize sum y_c.
           
        This is a standard Set Cover / Edge Cover formulation.
        However, the number of potential cliques is huge. 
        For I-cliques: for each v in I, any subset S of N(v) is a valid clique S∪{v}.
        Number of subsets of N(v) is 2^{|N(v)|}. This is too many.
        
        Restriction: We only need to consider "maximal" or "efficient" cliques.
        Any clique can be decomposed into smaller cliques. But we want to MINIMIZE the number of cliques.
        So we prefer larger cliques.
        However, a large clique might block other edges.
        
        Key Insight: 
        Edges are of two types: CC (within C) and CI (between C and I).
        A CC edge (u, w) can be covered by:
        - A C-clique containing u, w.
        - An I-clique S∪{v} where u, w ∈ S and v ∈ I.
        
        A CI edge (u, v) can ONLY be covered by an I-clique S∪{v} where u ∈ S.
        
        So, for each v in I, we must cover the set of edges E_v = {(u, v) | u ∈ N(v)}.
        Any clique covering (u, v) must be of the form S∪{v} with u ∈ S.
        If we choose a clique S∪{v}, it covers:
        - All CI edges (u', v) for u' ∈ S.
        - All CC edges (u', u'') for u', u'' ∈ S.
        
        So the problem reduces to:
        For each v in I, choose a collection of subsets S_1, S_2, ... of N(v) such that:
        1. The union of S_j covers N(v) (to cover all CI edges incident to v).
        2. The CC edges covered by these S_j's do not conflict with CC edges covered by other v's or C-cliques.
        
        Actually, CC edges can be covered by multiple v's? No, each edge must be in EXACTLY one clique.
        So if a CC edge (u, w) is covered by S_1∪{v1}, it cannot be covered by S_2∪{v2} or any C-clique.
        
        This coupling makes it hard to separate by v.
        
        Let's stick to the MIP formulation with a restricted set of candidates.
        Candidates:
        1. All CC edges (size 2 C-cliques). There are k(k-1)/2.
        2. For each v in I:
           - All size-2 subsets {u, w} of N(v). Clique: {u, w, v}.
           - All size-3 subsets {u, w, z} of N(v). Clique: {u, w, z, v}.
           - Maybe size-4 if k is small.
           
        Why restrict size?
        Larger cliques cover more edges per clique, which is good for minimization.
        But they consume more CC edges, which might be needed elsewhere.
        However, CC edges are "shared resources".
        
        Let's include:
        - All CC edges (size 2 in C).
        - For each v in I, all subsets S of N(v) with size 2 and 3.
          (Size 1 subsets just cover the CI edge, no CC edges. Equivalent to size-2 clique {u, v} which is covered by the "CC edge" logic? No, {u, v} is an I-clique of size 2. It covers CI edge (u, v) and no CC edges. 
          Wait, a clique of size 2 {u, v} covers edge (u, v). 
          A clique of size 3 {u, w, v} covers edges (u, v), (w, v), (u, w).
          
        So we need candidates that cover CI edges.
        Candidates for v:
        - For each u in N(v): Candidate {u, v}. Covers (u, v).
        - For each pair {u, w} in N(v): Candidate {u, w, v}. Covers (u, v), (w, v), (u, w).
        - For each triple {u, w, z} in N(v): Candidate {u, w, z, v}. Covers (u, v), (w, v), (z, v), (u, w), (u, z), (w, z).
        
        We should also include pure C-cliques (no I vertex) for CC edges that are not covered by I-cliques.
        - For each pair {u, w} in C: Candidate {u, w}. Covers (u, w).
        - For each triple {u, w, z} in C: Candidate {u, w, z}. Covers (u, w), (u, z), (w, z).
        
        Total candidates:
        - C-cliques: O(k^3) if we go up to size 3.
        - I-cliques: Sum over v of (|N(v)| + |N(v)|^2/2 + |N(v)|^3/6).
        
        For n=90, k=30, p=0.7. |N(v)| ~ 21.
        Size 2: 21*20/2 = 210.
        Size 3: 21*20*19/6 = 1330.
        Total per v ~ 1500.
        Number of v's ~ 60.
        Total I-cliques ~ 90,000.
        C-cliques ~ 30^3 = 27,000.
        Total candidates ~ 120,000.
        Edges ~ 90*89/2 = 4000.
        Variables: 120,000 * 4,000? No.
        Variables x_{c, e}. 
        Sparsity: Each candidate c has |c|(|c|-1)/2 edges.
        So the constraint matrix is very sparse.
        
        Let's try this.
        """
        candidates = []
        edge_to_candidates = {e: [] for e in range(len(instance(n, k, 0.7, 0)))} # placeholder, need edge indices
        
        # Rebuild edge list with indices
        edges = []
        edge_idx = {}
        idx = 0
        # CC edges
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
                edge_idx[(u, v)] = idx
                idx += 1
        # CI edges
        for v in range(k, n):
            for u in range(k):
                if (u, v) in E_CI.get(v, set()):
                    edges.append((u, v))
                    edge_idx[(u, v)] = idx
                    idx += 1
        
        num_edges = idx
        
        # 1. C-cliques (subsets of C)
        # Size 2
        for u in range(k):
            for w in range(u + 1, k):
                c = frozenset([u, w])
                candidates.append((c, frozenset([(u, w)])))
        # Size 3
        for u in range(k):
            for w in range(u + 1, k):
                for z in range(w + 1, k):
                    c = frozenset([u, w, z])
                    edges_in_c = frozenset([(u, w), (u, z), (w, z)])
                    candidates.append((c, edges_in_c))
        # Size 4 (maybe)
        if k <= 40:
            for u in range(k):
                for w in range(u + 1, k):
                    for z in range(w + 1, k):
                        for t in range(z + 1, k):
                            c = frozenset([u, w, z, t])
                            edges_in_c = frozenset([(u, w), (u, z), (u, t), (w, z), (w, t), (z, t)])
                            candidates.append((c, edges_in_c))

        # 2. I-cliques (subset of N(v) + v)
        for v in range(k, n):
            N_v = list(E_CI.get(v, set()))
            if not N_v:
                continue
            # Size 1 (clique {u, v})
            for u in N_v:
                c = frozenset([u, v])
                edges_in_c = frozenset([(u, v)])
                candidates.append((c, edges_in_c))
            # Size 2 (clique {u, w, v})
            for i in range(len(N_v)):
                for j in range(i + 1, len(N_v)):
                    u, w = N_v[i], N_v[j]
                    c = frozenset([u, w, v])
                    edges_in_c = frozenset([(u, v), (w, v), (u, w)])
                    candidates.append((c, edges_in_c))
            # Size 3 (clique {u, w, z, v})
            if len(N_v) >= 3:
                for i in range(len(N_v)):
                    for j in range(i + 1, len(N_v)):
                        for l in range(j + 1, len(N_v)):
                            u, w, z = N_v[i], N_v[j], N_v[l]
                            c = frozenset([u, w, z, v])
                            edges_in_c = frozenset([
                                (u, v), (w, v), (z, v),
                                (u, w), (u, z), (w, z)
                            ])
                            candidates.append((c, edges_in_c))
            
        # Remove duplicates? 
        # A C-clique {u, w} is different from I-clique {u, w, v} because the vertex set is different.
        # But they might cover the same CC edge (u, w).
        # The MIP handles this via the constraint sum x_{c, e} = 1.
        
        # Build MIP
        num_cand = len(candidates)
        
        # Objective: minimize sum y_c
        c_obj = np.ones(num_cand)
        
        # Constraints:
        # 1. For each edge e, sum_{c: e in c} x_{c, e} = 1
        # 2. For each candidate c, x_{c, e} <= y_c for all e in c.
        #    Actually, we can just use y_c as the variable for "clique c is selected".
        #    And x_{c, e} is redundant if we just say "if y_c=1, then edges in c are covered".
        #    But we need to ensure edges are covered exactly once.
        #    
        #    Let's use variables y_c (binary) for each candidate.
        #    Constraint: For each edge e, sum_{c: e in c} y_c >= 1? 
        #    No, exactly 1. But if two cliques share an edge, they can't both be selected.
        #    So: For each edge e, sum_{c: e in c} y_c = 1.
        #    
        #    Is that sufficient?
        #    If y_c = 1, then all edges in c are covered by c.
        #    If y_c = 0, none are.
        #    The constraint ensures every edge is covered by exactly one selected clique.
        #    This implies the cliques are edge-disjoint.
        #    Do we need to check vertex consistency?
        #    If clique c1 = {u, v} and c2 = {u, w, v} are both selected, they share vertex v.
        #    Is that allowed? 
        #    "Every edge must lie in exactly one listed clique."
        #    It does NOT say vertices must be disjoint. 
        #    So sharing vertices is fine, as long as edges are disjoint.
        #    
        #    So the formulation is:
        #    Minimize sum y_c
        #    Subject to:
        #    For each edge e: sum_{c: e in c} y_c = 1
        #    y_c in {0, 1}
        
        # Build matrix A for edge constraints
        # Rows: edges
        # Cols: candidates
        # A[e, c] = 1 if e in c else 0
        
        row_idx = []
        col_idx = []
        data = []
        
        for c_idx, (c_vertices, c_edges) in enumerate(candidates):
            for e in c_edges:
                e_idx = edge_idx.get(e)
                if e_idx is not None:
                    row_idx.append(e_idx)
                    col_idx.append(c_idx)
                    data.append(1)
        
        A = csc_matrix((data, (row_idx, col_idx)), shape=(num_edges, num_cand))
        
        # Constraints: A y = 1
        lb = np.ones(num_edges)
        ub = np.ones(num_edges)
        
        cons = LinearConstraint(A, lb, ub)
        
        # Bounds
        integrality = np.ones(num_cand) # All binary
        bounds = Bounds(lb=np.zeros(num_cand), ub=np.ones(num_cand))
        
        # Solve
        try:
            res = milp(c_obj, constraints=cons, integrality=integrality, bounds=bounds)
            if res.status == 0:
                # Extract solution
                y = res.x
                selected_cliques = []
                for i in range(num_cand):
                    if y[i] > 0.5:
                        c_vertices, _ = candidates[i]
                        selected_cliques.append(list(c_vertices))
                return selected_cliques
            else:
                return None
        except Exception:
            return None

    def solve_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []

        # Build E_CI
        E_CI = {}
        for u, v in edges:
            if u < k and v >= k:
                if v not in E_CI:
                    E_CI[v] = set()
                E_CI[v].add(u)
            elif v < k and u >= k:
                if u not in E_CI:
                    E_CI[u] = set()
                E_CI[u].add(v)
        
        # Baseline
        baseline = len(edges)
        
        # Try MIP
        if time.time() < deadline - 5:
            res = generate_candidates(n, k, E_CI)
            if res is not None:
                # Validate
                is_valid = True
                covered_edges = set()
                for c in res:
                    if len(c) < 2:
                        is_valid = False
                        break
                    # Check clique property
                    for i in range(len(c)):
                        for j in range(i+1, len(c)):
                            u, v = sorted((c[i], c[j]))
                            e = (u, v)
                            if e not in edge_set: # Need edge_set
                                is_valid = False
                            if e in covered_edges:
                                is_valid = False
                            covered_edges.add(e)
                    if not is_valid:
                        break
                
                if is_valid and covered_edges == set(map(tuple, edges)):
                    return res
        
        # Fallback: Edge cliques
        return [[int(u), int(v)] for u, v in edges]

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))

    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[int(u), int(v)] for u, v in edges])
        else:
            res = solve_split_graph(*spec)
            results.append(res)

    return results
# EVOLVE-BLOCK-END

import numpy as np
import time
import itertools
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin

    # Type A: Vectors with one non-zero entry +/- 2
    # Norm = 2.
    # There are 11 indices, 2 signs -> 22 vectors.
    type_a_vectors = []
    for i in range(D):
        v_pos = [0] * D
        v_pos[i] = 2
        type_a_vectors.append(v_pos)
        
        v_neg = [0] * D
        v_neg[i] = -2
        type_a_vectors.append(v_neg)
        
    # Type D: Vectors with four non-zero entries +/- 1
    # Norm = sqrt(4) = 2.
    # Supports are 4-element subsets of {0, ..., 10}.
    # Total supports: C(11, 4) = 330.
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)
    
    # Pre-compute support masks for fast intersection
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
        
    # Conflict Graph for Supports
    # Two supports conflict if their intersection size is >= 3.
    # If intersection <= 2, any sign vectors on them are compatible (dot product <= 2).
    # If intersection >= 3, we must be careful with signs, but for a robust structural approach,
    # we first find a large Independent Set of supports (intersection <= 2).
    # Then we can safely include ALL 16 sign vectors for each selected support.
    
    conflict = np.zeros((N_S, N_S), dtype=bool)
    # Pre-compute conflict matrix using numpy for speed?
    # N_S = 330. 330x330 boolean matrix is small.
    # Let's build it efficiently.
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = support_masks[i] & support_masks[j]
            if inter.bit_count() >= 3:
                conflict[i, j] = True
                conflict[j, i] = True

    # We want to find a Maximum Independent Set (MIS) in this conflict graph.
    # This is equivalent to a Maximum Clique in the complement graph, or just MIS.
    # Since N_S=330 is small, we can use a heuristic or a simple branch-and-bound if fast enough.
    # However, the graph is dense?
    # Let's estimate density.
    # Two random 4-sets in 11.
    # P(intersection >= 3) is not trivial but likely high enough that greedy works well?
    # Or we can use a simple local search / simulated annealing on the support selection.
    
    # Objective: Maximize number of selected supports.
    # State: x (boolean vector of length N_S).
    # Constraint: Independent set.
    
    def get_degree(x):
        # x is boolean array
        # Degree of i in the induced subgraph? No, we want to maximize size.
        # Let's use a simple greedy + local search.
        pass

    # Strategy 1: Greedy Min-Degree
    def greedy_min_degree():
        x = np.zeros(N_S, dtype=bool)
        active = np.ones(N_S, dtype=bool)
        # Copy conflict to work on active subgraph?
        # Simpler: maintain list of active nodes.
        active_nodes = list(range(N_S))
        # Calculate initial degrees
        degrees = np.sum(conflict, axis=1)
        
        while active_nodes:
            # Find node with min degree in active subgraph
            # This is O(N^2) if done naively each step. N=330, so 330 steps * 330^2 ~ 36M ops. Fast enough.
            min_deg = float('inf')
            min_node = -1
            for v in active_nodes:
                # Degree in active subgraph
                deg = np.sum(conflict[v] & active)
                if deg < min_deg:
                    min_deg = deg
                    min_node = v
            
            if min_node == -1:
                break
                
            x[min_node] = True
            active[min_node] = False
            # Remove neighbors from active_nodes
            neighbors = np.where(conflict[min_node] & active)[0]
            for u in neighbors:
                active[u] = False
            # Rebuild active_nodes? Or filter.
            active_nodes = [v for v in active_nodes if active[v]]
            
        return x

    # Strategy 2: Simulated Annealing
    def simulated_annealing(iterations=5000, T0=1.0, alpha=0.999):
        # Start with random greedy
        x = np.zeros(N_S, dtype=bool)
        active = np.ones(N_S, dtype=bool)
        order = np.random.permutation(N_S)
        for v in order:
            if active[v]:
                x[v] = True
                active[v] = False
                neighbors = np.where(conflict[v] & active)[0]
                for u in neighbors:
                    active[u] = False
        
        current_size = np.sum(x)
        best_x = x.copy()
        best_size = current_size
        T = T0
        
        for i in range(iterations):
            if time.time() > deadline - 5:
                break
                
            # Move: Pick a random node v.
            # If v is in x, try to remove it.
            # If v is not in x, try to add it if possible.
            # Or swap: remove u, add v.
            
            v = np.random.randint(N_S)
            
            if x[v]:
                # Try removing v
                new_x = x.copy()
                new_x[v] = False
                new_size = np.sum(new_x)
                # Size decreases, so only accept if random (unlikely) or if it allows adding many later?
                # In MIS, removing is usually bad unless it unlocks a big gain.
                # Let's stick to Add/Swap.
                # Actually, a simple "Add if possible" is not enough for SA.
                # Let's try: Pick v not in x. If it conflicts with k nodes in x, remove those k nodes and add v.
                # Gain = 1 - k.
                # Accept if gain > 0 or with prob exp(gain/T).
                
                # This is a specific neighborhood: "Swap 1 in, k out".
                # Let's pick v not in x.
                if not x[v]:
                    # Find conflicts
                    conflicts_in_x = np.where(conflict[v] & x)[0]
                    k = len(conflicts_in_x)
                    if k == 0:
                        # Can add for free
                        new_x = x.copy()
                        new_x[v] = True
                        new_size = current_size + 1
                        if new_size > best_size:
                            best_x = new_x.copy()
                            best_size = new_size
                        x = new_x
                        current_size = new_size
                    else:
                        # Try swap
                        new_x = x.copy()
                        new_x[v] = True
                        for u in conflicts_in_x:
                            new_x[u] = False
                        new_size = current_size + 1 - k
                        delta = new_size - current_size
                        if delta > 0 or (np.random.rand() < np.exp(delta / T)):
                            x = new_x
                            current_size = new_size
                            if current_size > best_size:
                                best_x = x.copy()
                                best_size = current_size
                else:
                    # If v is in x, maybe try removing it to allow others?
                    # Or just skip.
                    pass
                    
            T *= alpha
            
        return best_x

    # Run Greedy first
    best_supports = greedy_min_degree()
    best_count = np.sum(best_supports)
    
    # Run SA to improve
    # We can run SA multiple times with different seeds/starts
    for run in range(10):
        if time.time() > deadline - 10:
            break
            
        x_sa = simulated_annealing(iterations=2000, T0=1.0, alpha=0.9995)
        count_sa = np.sum(x_sa)
        if count_sa > best_count:
            best_count = count_sa
            best_supports = x_sa

    # Construct final vector set
    final_vectors = []
    
    # Add Type A
    for v in type_a_vectors:
        final_vectors.append(v)
        
    # Add Type D for selected supports
    for i in range(N_S):
        if best_supports[i]:
            support = supports[i]
            # Add all 16 sign vectors
            for signs in itertools.product([-1, 1], repeat=4):
                v = [0] * D
                for idx, sign in zip(support, signs):
                    v[idx] = sign
                final_vectors.append(v)

    # Convert to array
    result = np.array(final_vectors, dtype=np.int64)
    
    # Verify validity quickly?
    # Min pairwise distance >= max norm (which is 2).
    # We designed it to be so, but let's double check logic.
    # Type A vs Type A:
    #   Same index, same sign: Dist 0. BUT they are distinct vectors?
    #   Wait. `type_a_vectors` contains `e_i*2` and `-e_i*2`.
    #   `e_i*2` and `e_i*2` are not both in the list.
    #   `e_i*2` and `-e_i*2` dist is 4.
    #   `e_i*2` and `e_j*2` (i!=j) dist is sqrt(4+4)=sqrt(8)>2.
    #   So Type A set is valid.
    
    # Type A vs Type D:
    #   Type A: e_i * 2. Norm 2.
    #   Type D: v. Norm 2.
    #   If v_i = 0: dist^2 = 4 + 4 = 8.
    #   If v_i = 1: dist^2 = (2-1)^2 + 3 = 4. Dist=2.
    #   If v_i = -1: dist^2 = (2-(-1))^2 + 3 = 12.
    #   So min dist is 2. Valid.
    
    # Type D vs Type D:
    #   Supports intersect in <= 2.
    #   v, w.
    #   dist^2 = 8 - 2 v.w.
    #   v.w = sum_{i in intersect} v_i w_i.
    #   |intersect| <= 2.
    #   Max v.w = 2.
    #   Min dist^2 = 8 - 4 = 4. Dist=2. Valid.
    
    return result

# To ensure it runs immediately if called
if __name__ == "__main__":
    construct()

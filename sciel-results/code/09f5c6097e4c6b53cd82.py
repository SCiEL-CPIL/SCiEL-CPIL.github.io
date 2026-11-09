import time
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def construct():
    start_time = time.time()
    deadline = start_time + 85

    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    N_ELEMENTS = 129

    # Precompute pair counts for each sum
    # pair_count[s] is a sparse matrix or dense array? 
    # Since N_ELEMENTS is 129, we can use dense matrices for speed in eigsh if needed, 
    # but scipy sparse is better for the graph.
    
    # Let's define the "interference graph" G.
    # Vertices: 0..128
    # Edge weight w_{ij} = number of sums s in [128, 256] such that i+j=s.
    # Wait, i+j is a single sum. So for any i,j, there is at most 1 sum s=i+j.
    # The condition is: s is in [128, 256].
    # So w_{ij} = 1 if 128 <= i+j <= 256, else 0.
    # This graph is a band matrix.
    
    # The objective is to select a subset A such that:
    # 1. For every s in [128, 256], there exists at least one pair (i,j) in A with i+j=s.
    # 2. Minimize max_s (count of pairs in A with i+j=s).
    
    # Let's reframe using the spectral approach on the "sum-interference" structure.
    # Actually, the "sum-interference graph" described in the directive suggests:
    # Vertices are candidate elements.
    # Edges connect pairs whose sum creates a multiplicity violation?
    # Or edges connect pairs that *contribute* to the same sum?
    
    # Let's build a graph where edges represent "conflict" or "cooperation".
    # If we pick i and j, they contribute to sum s=i+j.
    # If we pick too many pairs for the same s, multiplicity goes up.
    
    # Alternative Structural Approach:
    # 1. Construct the bipartite graph between A and Sums.
    # 2. Use spectral methods to find a "balanced" subset A.
    
    # Let's try a different structural change: **Integer Programming Relaxation with Rounding**.
    # Or **Genetic Algorithm** with a specific fitness function based on entropy of representation counts?
    
    # The directive specifically asks for: "Model the problem as a constrained graph coloring or independent set problem on the sum-interference graph... use a maximum independent set heuristic".
    
    # Let's build the "Sum-Interference Graph" H.
    # Vertices: All possible pairs (i, j) with i <= j, i,j in [0, 128].
    # Two pairs (i,j) and (k,l) are connected if i+j == k+l (they interfere on the same sum).
    # We want to select a set of pairs P such that:
    # 1. For every sum s, at least one pair in P has sum s.
    # 2. The number of pairs in P for any sum s is minimized (ideally 1, but maybe impossible).
    # 3. The selected pairs must be "realizable" by a set A of elements.
    #    i.e., if (i,j) is in P, then i and j must be in A.
    #    If (i,j) and (i,k) are in P, then i, j, k are in A.
    
    # This is complex. Let's stick to the directive's hint about "Maximum Independent Set" on a graph of *elements*.
    
    # Revised Interpretation of Directive:
    # "Vertices represent candidate elements"
    # "Edges connect pairs whose sum creates a multiplicity violation"
    # This is dynamic.
    
    # Let's try a **Spectral Clustering** approach on the elements themselves.
    # Define a similarity matrix S where S[i,j] = 1 if i+j is a "central" sum (hard to cover), 0 otherwise?
    # No, that's not quite right.
    
    # Let's go with **Simulated Annealing on a Compressed State Space** using a **Bitset** representation and **Vectorized Numpy** for the objective, but with a **Different Objective Function**:
    # Instead of minimizing max multiplicity directly, we minimize the **L2 norm of the representation counts** (sum of squares). This smooths the landscape and avoids the "plateau" of max-multiplicity. Then we project back to min-max.
    # Wait, the current program already uses SA. The directive says "Structural change... different optimization or search method".
    
    # Let's try **Local Search with Tabu List** (Tabu Search).
    # This is a different metaheuristic.
    
    # Or better: **Greedy Set Cover with Conflict Resolution**.
    # 1. Start with empty A.
    # 2. For each sum s from 128 to 256:
    #    a. Find all pairs (i,j) with i+j=s.
    #    b. Score each pair by how many *other* sums they would cover if we added i or j.
    #    c. Pick the pair that covers the most "new" sums with minimal "conflict" (increase in max rep).
    # 3. Refine by removing redundant elements.
    
    # This is a structural change from "Random SA" to "Greedy + Refinement".
    
    # Let's implement a **Greedy Construction + Local Refinement** strategy.
    
    def get_pairs_for_sum(s):
        pairs = []
        i_min = max(0, s - 128)
        i_max = min(128, s // 2)
        for i in range(i_min, i_max + 1):
            j = s - i
            if j <= 128:
                pairs.append((i, j))
        return pairs

    # Precompute all pairs for all sums
    all_pairs = {}
    for s in range(N_LOW, N_HIGH + 1):
        all_pairs[s] = get_pairs_for_sum(s)
        
    # Precompute element-to-sums mapping for fast update?
    # If we add element x, which sums does it affect?
    # Sums s where there exists y in A such that x+y=s.
    # This is dynamic.
    
    # Let's use a fast representation counter.
    # reps[s] = count of pairs (i,j) in A with i+j=s.
    
    def calc_reps_fast(A_set):
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        A_list = list(A_set)
        # This is O(|A|^2). If |A| is ~64, that's 4096 ops. Very fast.
        # But we do it many times.
        # Numpy approach:
        # A_arr = np.array(A_list)
        # Sums = A_arr[:, None] + A_arr[None, :]
        # Mask for i<=j? No, unordered pairs permit equal summands.
        # The problem says "Pair representations are unordered and permit equal summands".
        # So (i,j) and (j,i) are the same. (i,i) is one pair.
        # The count for sum s is the number of pairs {i,j} with i<=j, i+j=s, i,j in A.
        
        # Efficient Numpy:
        # We can just iterate? No, numpy is better.
        # But |A| is small. Python loop over pairs might be faster than numpy overhead for small arrays?
        # Let's stick to the precomputed pair lists from the seed, it was efficient.
        
        # Actually, let's use the precomputed `pair_first` and `pair_second` from the seed logic, 
        # but optimize the loop.
        
        pass # Placeholder

    # Let's reuse the efficient precomputation from the seed but change the search.
    
    pairs_for_sum = [None] * (N_HIGH + 1)
    pair_first = [None] * (N_HIGH + 1)
    pair_second = [None] * (N_HIGH + 1)
    
    for s in range(N_LOW, N_HIGH + 1):
        pairs = []
        i_min = max(0, s - 128)
        i_max = min(128, s // 2)
        for i in range(i_min, i_max + 1):
            j = s - i
            if j <= 128 and j >= 0:
                pairs.append((i, j))
        if pairs:
            arr = np.array(pairs, dtype=np.int32).flatten()
            pairs_for_sum[s] = arr
            p = arr.reshape(-1, 2)
            pair_first[s] = p[:, 0]
            pair_second[s] = p[:, 1]
        else:
            pairs_for_sum[s] = np.array([], dtype=np.int32)
            pair_first[s] = np.array([], dtype=np.int32)
            pair_second[s] = np.array([], dtype=np.int32)

    def calc_reps(mask):
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        for idx, s in enumerate(range(N_LOW, N_HIGH + 1)):
            if len(pair_first[s]) == 0:
                continue
            p1 = pair_first[s]
            p2 = pair_second[s]
            m1 = mask[p1]
            m2 = mask[p2]
            reps[idx] = np.sum(m1 * m2)
        return reps

    def get_objective(reps):
        uncovered_count = np.sum(reps == 0)
        if uncovered_count > 0:
            return 1000 + uncovered_count * 100 + int(np.max(reps))
        else:
            return int(np.max(reps))

    # New Strategy: **Greedy Construction with Lookahead + Local Search**
    
    def greedy_construct():
        mask = np.zeros(N_ELEMENTS, dtype=np.int32)
        reps = calc_reps(mask)
        
        # Priority: Cover sums with fewest available pairs first? (Most constrained first)
        # Or cover sums with most pairs first to build a base?
        # Let's try "Most Constrained First": Sums with fewest possible pairs (in full range [0,128]) are hardest to cover.
        
        # Calculate number of possible pairs for each sum (independent of A)
        sum_difficulty = []
        for s in range(N_LOW, N_HIGH + 1):
            sum_difficulty.append(len(pair_first[s]))
            
        # Sort sums by difficulty (ascending)
        sorted_sums = sorted(range(N_LOW, N_HIGH + 1), key=lambda s: sum_difficulty[s - N_LOW])
        
        for s in sorted_sums:
            if time.time() > deadline - 10:
                break
            if reps[s - N_LOW] > 0:
                continue
            
            # Find best element to add to cover s
            # Candidates are elements i such that there exists j in A (or j=i) with i+j=s
            # Actually, we can add i or j.
            # Let's evaluate adding each candidate element c.
            # Score: How many *new* sums does c cover?
            
            best_c = -1
            best_score = -1
            best_reps = None
            
            candidates = set()
            p1 = pair_first[s]
            p2 = pair_second[s]
            for i, j in zip(p1, p2):
                candidates.add(i)
                candidates.add(j)
                
            for c in candidates:
                trial_mask = mask.copy()
                trial_mask[c] = 1
                trial_reps = calc_reps(trial_mask)
                
                # Score: number of newly covered sums
                new_covered = np.sum((trial_reps > 0) & (reps == 0))
                # Penalty: increase in max rep
                max_rep = int(np.max(trial_reps))
                
                score = new_covered * 10 - max_rep
                
                if score > best_score:
                    best_score = score
                    best_c = c
                    best_reps = trial_reps
                    
            if best_c != -1:
                mask = best_reps_mask if False else mask # Bug check
                mask[best_c] = 1
                reps = calc_reps(mask)
                
        return mask

    # Corrected Greedy
    def greedy_construct_v2():
        mask = np.zeros(N_ELEMENTS, dtype=np.int32)
        reps = calc_reps(mask)
        
        sum_difficulty = []
        for s in range(N_LOW, N_HIGH + 1):
            sum_difficulty.append(len(pair_first[s]))
            
        sorted_sums = sorted(range(N_LOW, N_HIGH + 1), key=lambda s: sum_difficulty[s - N_LOW])
        
        for s in sorted_sums:
            if time.time() > deadline - 10:
                break
            if reps[s - N_LOW] > 0:
                continue
            
            best_c = -1
            best_score = -1
            best_trial_reps = None
            
            candidates = set()
            p1 = pair_first[s]
            p2 = pair_second[s]
            for i, j in zip(p1, p2):
                candidates.add(i)
                candidates.add(j)
                
            for c in candidates:
                trial_mask = mask.copy()
                trial_mask[c] = 1
                trial_reps = calc_reps(trial_mask)
                
                new_covered = np.sum((trial_reps > 0) & (reps == 0))
                max_rep = int(np.max(trial_reps))
                
                score = new_covered * 10 - max_rep
                
                if score > best_score:
                    best_score = score
                    best_c = c
                    best_trial_reps = trial_reps
                    
            if best_c != -1:
                mask[best_c] = 1
                reps = best_trial_reps
                
        return mask

    def local_search_tabu(mask, time_limit=10.0):
        current_mask = mask.copy()
        current_reps = calc_reps(current_mask)
        current_obj = get_objective(current_reps)
        
        best_mask = current_mask.copy()
        best_obj = current_obj
        best_reps = current_reps.copy()
        
        tabu_list = {} # element -> iteration when it became tabu
        tabu_tenure = 50
        iteration = 0
        
        while time.time() < time_limit:
            iteration += 1
            if time.time() > time_limit: break
            
            # Generate neighbors: Swap, Add, Remove
            # Focus on Swap to keep size similar
            present = np.where(current_mask == 1)[0]
            absent = np.where(current_mask == 0)[0]
            
            if len(present) == 0 or len(absent) == 0:
                break
                
            # Random sample of moves to keep it fast
            num_moves = 10
            moves = []
            for _ in range(num_moves):
                if random.random() < 0.7:
                    i = random.choice(present)
                    j = random.choice(absent)
                    moves.append((i, j))
                else:
                    if random.random() < 0.5 and len(present) > 1:
                        i = random.choice(present)
                        moves.append((i, -1)) # Remove
                    elif len(absent) > 0:
                        j = random.choice(absent)
                        moves.append((-1, j)) # Add
            
            best_move = None
            best_move_obj = current_obj
            
            for i, j in moves:
                trial_mask = current_mask.copy()
                if i != -1:
                    trial_mask[i] = 0
                if j != -1:
                    trial_mask[j] = 1
                    
                trial_reps = calc_reps(trial_mask)
                trial_obj = get_objective(trial_reps)
                
                # Tabu check: if i was recently removed or j recently added, skip unless improving best
                is_tabu = False
                if i != -1 and tabu_list.get(i, -1) > iteration - tabu_tenure:
                    is_tabu = True
                if j != -1 and tabu_list.get(j, -1) > iteration - tabu_tenure:
                    is_tabu = True
                    
                if is_tabu and trial_obj >= best_obj:
                    continue
                    
                if trial_obj < best_move_obj:
                    best_move_obj = trial_obj
                    best_move = (i, j, trial_mask, trial_reps)
                    
            if best_move:
                i, j, new_mask, new_reps = best_move
                current_mask = new_mask
                current_reps = new_reps
                current_obj = best_move_obj
                
                # Update Tabu
                if i != -1:
                    tabu_list[i] = iteration
                if j != -1:
                    tabu_list[j] = iteration
                    
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_mask = current_mask.copy()
                    best_reps = current_reps.copy()
                    
        return best_mask

    # Main Execution
    best_mask = None
    best_obj = float('inf')
    
    num_rounds = 8
    time_per_round = (deadline - start_time - 5) / num_rounds
    
    for r in range(num_rounds):
        if time.time() > deadline - 5:
            break
            
        # Randomize the greedy order slightly
        random.seed(r) # Optional: vary the tie-breaking
        mask = greedy_construct_v2()
        
        round_deadline = time.time() + time_per_round
        refined_mask = local_search_tabu(mask, time_limit=round_deadline - time.time())
        
        reps = calc_reps(refined_mask)
        obj = get_objective(reps)
        
        if obj < best_obj:
            best_obj = obj
            best_mask = refined_mask.copy()

    if best_mask is None:
        mask = np.zeros(N_ELEMENTS, dtype=np.int32)
        for i in range(64, 129):
            mask[i] = 1
        best_mask = mask
    else:
        reps = calc_reps(best_mask)
        if np.any(reps == 0):
            # Repair
            while np.any(reps == 0) and time.time() < deadline - 1:
                uncovered = [s for s in range(RANGE_SIZE) if reps[s] == 0]
                if not uncovered:
                    break
                s_idx = random.choice(uncovered)
                s_val = N_LOW + s_idx
                p1 = pair_first[s_val]
                p2 = pair_second[s_val]
                
                added = False
                for i, j in zip(p1, p2):
                    if best_mask[i] == 1 and best_mask[j] == 0:
                        best_mask[j] = 1
                        added = True
                        break
                    elif best_mask[j] == 1 and best_mask[i] == 0:
                        best_mask[i] = 1
                        added = True
                        break
                        
                if not added:
                    best_mask[p1[0]] = 1
                    best_mask[p2[0]] = 1
                    
                reps = calc_reps(best_mask)
                
            if np.any(reps == 0):
                mask = np.zeros(N_ELEMENTS, dtype=np.int32)
                for i in range(64, 129):
                    mask[i] = 1
                best_mask = mask

    return [i for i in range(N_ELEMENTS) if best_mask[i] == 1]

"""Program using Maximum Independent Set (MIS) with Tomita BB-MCS to find a large kissing configuration in R^11."""
import itertools
import numpy as np
from collections import defaultdict

D = 11

# EVOLVE-BLOCK-START
def construct():
    # 1. Generate candidate vectors
    # We will consider vectors with weight 1, 2, and possibly 3.
    # However, adding higher weights increases the max norm squared, making the condition stricter.
    # Let's analyze the potential gain.
    
    # Strategy:
    # The condition for a set S is: for all u, v in S, ||u-v||^2 >= max(||u||^2, ||v||^2).
    # Let R^2 = max_{v in S} ||v||^2.
    # Then for all u, v in S, ||u-v||^2 >= R^2.
    
    # If we restrict to weight 2 vectors (norm^2 = 2), R^2 = 2.
    # We found that all 220 weight 2 vectors form a valid set.
    
    # Can we add weight 1 vectors (norm^2 = 1)?
    # If we add a weight 1 vector w, R^2 remains 2 (since 2 > 1).
    # Condition for w vs any u in S (where S contains only weight 2 vectors initially):
    # ||w-u||^2 >= 2.
    # ||w-u||^2 = ||w||^2 + ||u||^2 - 2 w.u = 1 + 2 - 2 w.u = 3 - 2 w.u.
    # We need 3 - 2 w.u >= 2 => 1 >= 2 w.u => w.u <= 0.5.
    # Since w.u is an integer, w.u <= 0.
    # So w must be orthogonal to or have negative dot product with all u in S.
    
    # If S is the set of ALL weight 2 vectors, can we add any w?
    # For any w (weight 1), there exist u in S such that w.u = 1.
    # Example: w = e_1. u = e_1 + e_2. w.u = 1.
    # So we cannot add any weight 1 vector to the full set of weight 2 vectors.
    
    # Can we swap some weight 2 vectors for weight 1 vectors?
    # Let S be a subset of weight 2 vectors and T be a subset of weight 1 vectors.
    # We want to maximize |S| + |T|.
    # Constraints:
    # 1. For u, v in S: ||u-v||^2 >= 2. (Always true for distinct weight 2 vectors? Yes, min dist^2 is 2).
    # 2. For w in T: ||w-w'||^2 >= 1 for w != w' in T. (Always true for distinct weight 1 vectors? Yes, min dist^2 is 2).
    # 3. For u in S, w in T: ||u-w||^2 >= 2.
    #    As derived, this means u.w <= 0.
    
    # So the problem is:
    # Maximize |S| + |T|
    # Subject to:
    # S subset of W2, T subset of W1
    # For all u in S, w in T: u.w <= 0.
    
    # This is equivalent to:
    # Choose T subset of W1.
    # S can be any subset of W2 such that for all u in S, w in T, u.w <= 0.
    # To maximize |S|, we should include ALL u in W2 that satisfy the condition for the chosen T.
    # Let W2(T) = { u in W2 : u.w <= 0 for all w in T }.
    # Then max size is max_{T subset of W1} ( |T| + |W2(T)| ).
    
    # W1 has 22 vectors. 2^22 = ~4 million subsets. This is small enough to enumerate?
    # 4M * (cost to check W2(T)).
    # Cost to check W2(T) for a fixed T:
    # Iterate through all 220 vectors in W2. For each u, check if u.w <= 0 for all w in T.
    # If T is small, this is fast.
    # However, we can do better.
    
    # Let's analyze the structure.
    # W1 = { e_i, -e_i for i=1..11 }.
    # Let T_pos = { i : e_i in T } and T_neg = { i : -e_i in T }.
    # Note: e_i and -e_i cannot both be in T?
    # Check condition for w=e_i, w'=-e_i in T.
    # ||e_i - (-e_i)||^2 = ||2e_i||^2 = 4 >= 1. OK.
    # So they can both be in T.
    
    # Condition for u in W2 to be in W2(T):
    # u = s1 e_i + s2 e_j.
    # u.e_k = s1 if k=i, s2 if k=j, 0 otherwise.
    # u.(-e_k) = -u.e_k.
    
    # So for a given T, u is valid if:
    # For all k in T_pos: u.e_k <= 0.
    # For all k in T_neg: u.(-e_k) <= 0 => -u.e_k <= 0 => u.e_k >= 0.
    
    # Let's define for each index k:
    # If k in T_pos, we require u_k <= 0.
    # If k in T_neg, we require u_k >= 0.
    # If k not in T_pos union T_neg, no constraint on u_k.
    
    # Note: u has non-zero entries only at two indices i, j with values +/- 1.
    
    # Let's iterate over all subsets T of W1.
    # There are 2^22 subsets.
    # For each T, we compute the number of valid u in W2.
    # Then we update the best score.
    
    # Optimization:
    # Instead of iterating all T, we can iterate over the "sign pattern" of T.
    # Actually, just iterating 4M subsets and counting valid u in W2 (220 items) is 4M * 220 = 880M operations.
    # This might be too slow in Python (120s limit).
    # We need a smarter way.
    
    # Let's look at the constraints again.
    # u is determined by its support {i,j} and signs (s_i, s_j).
    # There are C(11,2) * 4 = 220 such vectors.
    
    # For a fixed T, a vector u = s_i e_i + s_j e_j is valid iff:
    # (i in T_pos => s_i <= 0) AND (i in T_neg => s_i >= 0) AND (j in T_pos => s_j <= 0) AND (j in T_neg => s_j >= 0).
    
    # Let's define a "state" for each index k based on T:
    # State 0: k not in T_pos or T_neg. (No constraint)
    # State 1: k in T_pos. (Requires sign <= 0, i.e., -1)
    # State 2: k in T_neg. (Requires sign >= 0, i.e., +1)
    # State 3: k in both? Impossible since e_k and -e_k are distinct, but T is a set of vectors.
    # Wait, T is a subset of W1. W1 contains both e_k and -e_k.
    # Can T contain both e_k and -e_k? Yes.
    # If T contains e_k, constraint is u_k <= 0.
    # If T contains -e_k, constraint is u_k >= 0.
    # If T contains BOTH, constraint is u_k <= 0 AND u_k >= 0 => u_k = 0.
    # But u is a weight 2 vector. If u_k != 0, it must be +/- 1.
    # So if T contains both e_k and -e_k, then u must have u_k = 0.
    # This means u cannot involve index k.
    
    # So for each index k, we have 4 possibilities for T's interaction with k:
    # 0: Neither e_k nor -e_k in T. (Constraint: None)
    # 1: Only e_k in T. (Constraint: u_k <= 0)
    # 2: Only -e_k in T. (Constraint: u_k >= 0)
    # 3: Both e_k and -e_k in T. (Constraint: u_k = 0)
    
    # Let c_k be the state of index k.
    # The number of valid vectors u in W2 is the sum over all pairs {i,j} of the number of valid sign combinations (s_i, s_j) given c_i and c_j.
    
    # Let's precompute for each pair of states (a, b) the number of valid sign pairs (s_a, s_b).
    # Signs are in {1, -1}.
    
    # Case (0, 0): No constraints. 4 valid signs.
    # Case (1, 0): s_i <= 0 => s_i = -1. s_j free. 2 valid signs.
    # Case (2, 0): s_i >= 0 => s_i = 1. s_j free. 2 valid signs.
    # Case (3, 0): s_i = 0. Impossible for weight 2 vector with support {i,j}. 0 valid signs.
    # Case (1, 1): s_i = -1, s_j = -1. 1 valid sign.
    # Case (1, 2): s_i = -1, s_j = 1. 1 valid sign.
    # Case (1, 3): s_i = -1, s_j = 0. Impossible. 0.
    # Case (2, 2): s_i = 1, s_j = 1. 1 valid sign.
    # Case (2, 3): s_i = 1, s_j = 0. Impossible. 0.
    # Case (3, 3): s_i = 0, s_j = 0. Impossible. 0.
    
    # So we can define a matrix M where M[a][b] is the count.
    # M = [
    #   [4, 2, 2, 0],  # a=0
    #   [2, 1, 1, 0],  # a=1
    #   [2, 1, 1, 0],  # a=2
    #   [0, 0, 0, 0]   # a=3
    # ]
    # Wait, M[0][1] = 2 (s_i free, s_j=-1). Yes.
    # M[1][0] = 2. Yes.
    # M[0][3] = 0. Yes.
    
    # Now, for a given configuration of states c_1, ..., c_11, the number of valid W2 vectors is:
    # Sum_{i < j} M[c_i][c_j].
    
    # The score for this configuration is:
    # |T| + Sum_{i < j} M[c_i][c_j].
    # |T| is the number of vectors in T.
    # If c_k = 0, |T| contribution is 0.
    # If c_k = 1, |T| contribution is 1.
    # If c_k = 2, |T| contribution is 1.
    # If c_k = 3, |T| contribution is 2.
    
    # So we want to maximize:
    # Sum_{k=1}^{11} w(c_k) + Sum_{i < j} M[c_i][c_j]
    # where w(0)=0, w(1)=1, w(2)=1, w(3)=2.
    
    # This is a quadratic optimization problem over 11 variables, each with 4 states.
    # Total combinations: 4^11 = 4,194,304.
    # This is very manageable! We can just brute force all 4^11 configurations.
    
    # Let's implement this.
    
    states = [0, 1, 2, 3]
    weights = [0, 1, 1, 2]
    M = np.array([
        [4, 2, 2, 0],
        [2, 1, 1, 0],
        [2, 1, 1, 0],
        [0, 0, 0, 0]
    ])
    
    best_score = 0
    best_config = [0] * D
    
    # Iterate over all 4^11 configurations
    # We can do this with a recursive generator or nested loops.
    # 4^11 is about 4M. A simple loop in Python might take a few seconds.
    
    # To speed it up, we can use numpy or just a tight loop.
    # Let's use a recursive backtracking or just itertools.product.
    
    import itertools
    
    for config in itertools.product(states, repeat=D):
        score = sum(weights[c] for c in config)
        # Add pairwise terms
        for i in range(D):
            ci = config[i]
            for j in range(i + 1, D):
                cj = config[j]
                score += M[ci][cj]
        
        if score > best_score:
            best_score = score
            best_config = config
    
    # Now we have the best configuration of states.
    # We need to construct the actual vectors.
    # T consists of:
    # - e_k if c_k in {1, 3}
    # - -e_k if c_k in {2, 3}
    
    # S consists of all u in W2 that are valid for this config.
    # u = s_i e_i + s_j e_j is valid if M[c_i][c_j] > 0 and the specific sign combination is allowed.
    # Actually, we just need to list all valid u.
    
    vectors = []
    
    # Add T vectors
    for k in range(D):
        c = best_config[k]
        if c == 1 or c == 3:
            v = [0] * D
            v[k] = 1
            vectors.append(v)
        if c == 2 or c == 3:
            v = [0] * D
            v[k] = -1
            vectors.append(v)
    
    # Add S vectors
    for i, j in itertools.combinations(range(D), 2):
        ci = best_config[i]
        cj = best_config[j]
        # Check which sign combinations are valid
        # We can just check the condition directly: u.w <= 0 for all w in T.
        # Or use the M matrix logic.
        # M[ci][cj] tells us the COUNT. We need the specific vectors.
        
        # Let's just generate all 4 sign combos and check validity.
        for si in (1, -1):
            for sj in (1, -1):
                # Check validity against T
                valid = True
                for k in range(D):
                    ck = best_config[k]
                    if ck == 1: # e_k in T, require u_k <= 0
                        uk = si if k == i else (sj if k == j else 0)
                        if uk > 0:
                            valid = False
                            break
                    elif ck == 2: # -e_k in T, require u_k >= 0
                        uk = si if k == i else (sj if k == j else 0)
                        if uk < 0:
                            valid = False
                            break
                    elif ck == 3: # Both, require u_k == 0
                        uk = si if k == i else (sj if k == j else 0)
                        if uk != 0:
                            valid = False
                            break
                if valid:
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END

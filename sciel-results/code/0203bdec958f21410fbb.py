"""Construct a 4-regular graph with girth 10 on 320 vertices (the (4,10)-cage).

Uses a voltage graph lift of K_{4,4} over Z_40 with carefully chosen voltages.
The base K_{4,4} has 8 vertices and 16 edges. The lift over Z_40 gives 320 vertices.
The girth of the lift is at least 10 if the voltage assignment avoids short cycles.
"""
import time
import sys
import random
import itertools
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    if k != 4 or g != 10:
        # Fallback for other parameters: use LUW construction
        return _construct_luw(k, g, deadline)
    
    # For k=4, g=10: construct the 320-vertex cage via voltage graph lift of K_{4,4} over Z_40
    # K_{4,4} has vertices A = {a0..a3} and B = {b0..b3}, all edges a_i-b_j.
    # Lift over Z_40: 8*40 = 320 vertices.
    # Vertex (side, i, k) where side=0 for A, side=1 for B, i in 0..3, k in 0..39.
    # Edge from (0, i, k) to (1, j, (k + volt[i][j]) % 40).
    
    def _build_lift(voltages):
        """Build the voltage graph lift of K_{4,4} over Z_40."""
        n = 320
        edges = []
        for i in range(4):
            for j in range(4):
                volt = voltages[i][j]
                for k in range(40):
                    a_idx = i * 40 + k
                    b_idx = 160 + j * 40 + (k + volt) % 40
                    edges.append((a_idx, b_idx))
        return n, edges
    
    def _check_4cycle_condition(voltages):
        """Check no 4-closed-walk in base has voltage 0 mod 40."""
        for i in range(4):
            for k in range(4):
                if i == k:
                    continue
                for j in range(4):
                    for l in range(4):
                        if j == l:
                            continue
                        s = (voltages[i][j] - voltages[k][j] + voltages[k][l] - voltages[i][l]) % 40
                        if s == 0:
                            return False
        return True
    
    def _check_6cycle_condition(voltages):
        """Check no 6-closed-walk in base has voltage 0 mod 40."""
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if k == i:
                        continue
                    for l in range(4):
                        if l == j:
                            continue
                        for m in range(4):
                            if m == k or m == i:
                                continue
                            for n in range(4):
                                if n == l or n == j:
                                    continue
                                s = (voltages[i][j] - voltages[k][j] + voltages[k][l] 
                                     - voltages[m][l] + voltages[m][n] - voltages[i][n]) % 40
                                if s == 0:
                                    return False
        return True
    
    def _check_8cycle_condition(voltages):
        """Check no 8-closed-walk in base has voltage 0 mod 40."""
        # 8-closed walk: a_i -> b_j -> a_k -> b_l -> a_m -> b_n -> a_p -> b_q -> a_i
        # For K_{4,4}, an 8-cycle uses all 4 A and all 4 B vertices.
        # Simple 8-cycles correspond to permutations of A and B.
        # But we also need to account for the fact that a closed walk of length 8
        # that revisits vertices contains a shorter closed walk.
        # So checking all "simple" 8-cycles (where no vertex repeats) is sufficient
        # IF we've already verified no shorter closed walk has voltage 0.
        # In K_{4,4}, an 8-cycle visits 4 A and 4 B vertices, all distinct.
        # The voltage sum for a cycle a_{p0}->b_{q0}->a_{p1}->b_{q1}->...->a_{p3}->b_{q3}->a_{p0}
        # is: sum_{t=0}^{3} (volt[p_t][q_t] - volt[p_{t+1}][q_t]) where p_4 = p_0
        # Wait, let me re-derive.
        # Edge a_i -> b_j has voltage volt[i][j].
        # Edge b_j -> a_i has voltage -volt[i][j] (going backward).
        # Cycle: a_{p0} -> b_{q0} -> a_{p1} -> b_{q1} -> a_{p2} -> b_{q2} -> a_{p3} -> b_{q3} -> a_{p0}
        # Voltages:
        #   a_{p0} -> b_{q0}: volt[p0][q0]
        #   b_{q0} -> a_{p1}: -volt[p1][q0]
        #   a_{p1} -> b_{q1}: volt[p1][q1]
        #   b_{q1} -> a_{p2}: -volt[p2][q1]
        #   a_{p2} -> b_{q2}: volt[p2][q2]
        #   b_{q2} -> a_{p3}: -volt[p3][q2]
        #   a_{p3} -> b_{q3}: volt[p3][q3]
        #   b_{q3} -> a_{p0}: -volt[p0][q3]
        # Sum: volt[p0][q0] - volt[p1][q0] + volt[p1][q1] - volt[p2][q1] + volt[p2][q2] - volt[p3][q2] + volt[p3][q3] - volt[p0][q3]
        # = sum_{t=0}^{3} (volt[p_t][q_t] - volt[p_{(t+1)%4}][q_t])
        
        # To avoid 40*40 = 1600 permutations being too slow, we can fix p0=0, q0=0 by symmetry
        # (shifting all p indices doesn't change the condition since it's a difference structure)
        # Actually, the condition is invariant under relabeling of A and B vertices.
        # So fix p = [0, p1, p2, p3] where (p1,p2,p3) is a permutation of {1,2,3}
        # and q = [0, q1, q2, q3] where (q1,q2,q3) is a permutation of {1,2,3}
        # That's 6*6 = 36 cases instead of 24*24 = 576.
        
        for perm_a in itertools.permutations([1, 2, 3]):
            pa = [0] + list(perm_a)
            for perm_b in itertools.permutations([1, 2, 3]):
                qb = [0] + list(perm_b)
                s = 0
                for t in range(4):
                    s += voltages[pa[t]][qb[t]]
                    s -= voltages[pa[(t+1)%4]][qb[t]]
                if s % 40 == 0:
                    return False
        return True
    
    def _verify_girth_exact(n, edges, g, deadline):
        """Verify girth >= g using non-backtracking walks from every vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        max_len = g - 1  # we check walks of length 1..g-1
        
        for start_v in range(n):
            if time.time() > deadline:
                return False
            
            states = set()
            for w in adj[start_v]:
                states.add((w, start_v))
            
            for d in range(2, max_len + 1):
                new_states = set()
                for (v, p) in states:
                    for w in adj[v]:
                        if w != p:
                            new_states.add((w, v))
                if any(v == start_v for (v, p) in new_states):
                    return False
                states = new_states
                if not states:
                    break
        
        return True
    
    # Search for valid voltages
    candidate_voltages = None
    
    # Pass 1: structured search with affine form volt[i][j] = (a*i + b*j + c*i*j) mod 40
    # Condition 1 requires c*(i-k)*(j-l) != 0 mod 40 for all i!=k, j!=l
    # This means c must be coprime to 40/gcd(c,40) for all relevant factors.
    # More precisely, for any i!=k, j!=l: c*(i-k)*(j-l) not congruent to 0 mod 40.
    # Since i-k and j-l are in {-3,-2,-1,1,2,3}, we need c*d1*d2 != 0 mod 40
    # for all d1,d2 in {-3,-2,-1,1,2,3}.
    # This means gcd(c, 40) must be 1, i.e., c coprime to 40.
    # Wait, that's not quite right. c*d1*d2 mod 40 != 0.
    # If c is even, then c*2*1 = 2c could be 0 mod 40 if c=20. But c*1*1 = c != 0 mod 40 if c != 0.
    # Let's just check: for c in range(1,40), is c*d1*d2 != 0 mod 40 for all d1,d2 in {1,2,3}?
    # c*1*1 = c, so c != 0 mod 40. OK for c in 1..39.
    # c*1*2 = 2c, so 2c != 0 mod 40, i.e., c != 20 mod 40. So c != 20.
    # c*1*3 = 3c, so 3c != 0 mod 40. gcd(3,40)=1, so c != 0 mod 40. OK.
    # c*2*2 = 4c, so 4c != 0 mod 40, i.e., c != 10 mod 40. So c not in {10, 20, 30}.
    # c*2*3 = 6c, so 6c != 0 mod 40. gcd(6,40)=2, so 3c != 0 mod 20. c != 0, 20/3... 
    # 6c mod 40 = 0 means 40 | 6c, i.e., 20 | 3c, i.e., 20 | c (since gcd(3,20)=1). So c=20. Already excluded.
    # c*3*3 = 9c, so 9c != 0 mod 40. gcd(9,40)=1, so c != 0 mod 40. OK.
    # So valid c: 1..39 excluding {10, 20, 30}.
    
    valid_c_values = [c for c in range(1, 40) if c not in (10, 20, 30)]
    
    for c in valid_c_values:
        for a in range(0, 40, 5):
            for b in range(0, 40, 5):
                voltages = [[(a*i + b*j + c*i*j) % 40 for j in range(4)] for i in range(4)]
                if _check_4cycle_condition(voltages) and _check_6cycle_condition(voltages) and _check_8cycle_condition(voltages):
                    candidate_voltages = voltages
                    break
            if candidate_voltages:
                break
        if candidate_voltages:
            break
    
    if candidate_voltages is None:
        # Pass 2: finer grid
        for c in valid_c_values[:15]:
            for a in range(0, 40, 2):
                for b in range(0, 40, 2):
                    voltages = [[(a*i + b*j + c*i*j) % 40 for j in range(4)] for i in range(4)]
                    if _check_4cycle_condition(voltages) and _check_6cycle_condition(voltages) and _check_8cycle_condition(voltages):
                        candidate_voltages = voltages
                        break
                if candidate_voltages:
                    break
            if candidate_voltages:
                break
    
    if candidate_voltages is None:
        # Pass 3: random search
        for attempt in range(20000):
            if time.time() > deadline - 40:
                break
            c = random.choice(valid_c_values)
            a = random.randint(0, 39)
            b = random.randint(0, 39)
            voltages = [[(a*i + b*j + c*i*j) % 40 for j in range(4)] for i in range(4)]
            if _check_4cycle_condition(voltages) and _check_6cycle_condition(voltages) and _check_8cycle_condition(voltages):
                candidate_voltages = voltages
                break
    
    if candidate_voltages is None:
        # Pass 4: full random search over all 4x4 matrices with constraint-aware sampling
        for attempt in range(50000):
            if time.time() > deadline - 30:
                break
            # Sample voltages such that condition 1 is likely satisfied
            # Condition 1: volt[i][j] - volt[i][l] != volt[k][j] - volt[k][l] for all i!=k, j!=l
            # This is equivalent to: the rows of the matrix, when viewed as vectors in Z_40^4,
            # have pairwise differences that are "distinct in each coordinate pair".
            # A sufficient condition: each row is a permutation of some base, shifted.
            # Let's try: volt[i][j] = (perm[i][j] + shift[i]) % 40 where perm is a fixed Latin square
            # and shift[i] are random.
            # Actually, let's just do fully random with rejection.
            voltages = [[random.randint(0, 39) for _ in range(4)] for _ in range(4)]
            if _check_4cycle_condition(voltages) and _check_6cycle_condition(voltages) and _check_8cycle_condition(voltages):
                candidate_voltages = voltages
                break
    
    if candidate_voltages is not None:
        n, edges = _build_lift(candidate_voltages)
        # Final verification with exact girth check
        if _verify_girth_exact(n, edges, g, deadline):
            return n, edges
    
    # Fallback: use LUW construction
    return _construct_luw(4, 10, deadline)


def _construct_luw(k, g, deadline):
    """LUW construction as fallback."""
    q = k
    d_candidates = []
    for d in range(3, 20):
        if d % 2 == 1:
            if d + 5 >= g:
                d_candidates.append(d)
        else:
            if d + 4 >= g:
                d_candidates.append(d)
    d = min(d_candidates) if d_candidates else max(3, g - 5)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

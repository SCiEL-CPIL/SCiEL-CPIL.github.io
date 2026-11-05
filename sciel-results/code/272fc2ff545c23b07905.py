import random
import time
import itertools
import math

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))
N_PTS = len(ALL_POINTS)
IDX_TO_POINT = {i: p for i, p in enumerate(ALL_POINTS)}
POINT_TO_IDX = {p: i for i, p in enumerate(ALL_POINTS)}

# Precompute isosceles triples (indices)
# A triple (i, j, k) is isosceles if dist(i,j)==dist(i,k) or dist(i,j)==dist(j,k) or dist(j,k)==dist(i,k)
# We need to ensure no such triple has all 3 points selected.
ISO_TRIPLES = []
for i in range(N_PTS):
    pi = ALL_POINTS[i]
    for j in range(i+1, N_PTS):
        pj = ALL_POINTS[j]
        d2_ij = (pi[0]-pj[0])**2 + (pi[1]-pj[1])**2
        for k in range(j+1, N_PTS):
            pk = ALL_POINTS[k]
            d2_ik = (pi[0]-pk[0])**2 + (pi[1]-pk[1])**2
            d2_jk = (pj[0]-pk[0])**2 + (pj[1]-pk[1])**2
            if d2_ij == d2_ik or d2_ij == d2_jk or d2_ik == d2_jk:
                ISO_TRIPLES.append((i, j, k))

# Precompute conflict neighbors for maximality
# For a point p to be "blocked" (cannot be added), there must exist i, j in S such that {p, i, j} is isosceles.
# This is equivalent to: p is in the "conflict set" of some pair (i, j) in S.
# Alternatively, for each point p, precompute the list of pairs (i, j) such that {p, i, j} is isosceles.
# If any such pair is fully selected, p is blocked.
# To check if S is maximal: for every p NOT in S, there must exist i, j in S such that (p, i, j) is isosceles.

# Let's precompute for each point p, the list of pairs (i, j) that form an isosceles triangle with p.
# Note: i, j can be any indices, not necessarily > p.
# To speed up, we can precompute a bitmask or a list of conflicting pairs for each point.
# However, checking maximality is the bottleneck.
# A point p is "addable" to S if for ALL i, j in S, {p, i, j} is NOT isosceles.
# Equivalently, p is "blocked" if there EXISTS i, j in S such that {p, i, j} IS isosceles.

# Precompute for each point p, the set of points q such that there exists r where {p, q, r} is isosceles?
# No, that's not quite right.
# Let's precompute for each point p, the list of pairs (i, j) such that {p, i, j} is isosceles.
# Then p is blocked if S contains any such pair (i, j).
# This is still O(|S|^2) per check if done naively.

# Better approach for Simulated Annealing:
# State: binary vector x of length N_PTS.
# Objective: Minimize sum(x)
# Constraints:
# 1. For all (i, j, k) in ISO_TRIPLES: NOT (x[i] and x[j] and x[k])
# 2. For all p NOT in S: EXISTS i, j in S such that {p, i, j} is isosceles.

# To make constraint 2 efficient, we can precompute for each p, the set of "witness pairs" W_p = {(i, j) | {p, i, j} is isosceles}.
# p is blocked if S intersects W_p in at least one pair. i.e., there is (i, j) in W_p such that x[i]=1 and x[j]=1.

# Precomputation:
# For each p, list of pairs (i, j) with i < j, such that {p, i, j} is isosceles.
# Note: i or j can be p? No, p is not in S, so i, j are in S, so i != p, j != p.
# So we only consider i, j != p.

WITNESS_PAIRS = [[] for _ in range(N_PTS)]
for i in range(N_PTS):
    for j in range(i+1, N_PTS):
        # Find all p such that {i, j, p} is isosceles
        # We can iterate k and check, but that's O(N^3) total.
        # N=16, N_PTS=256. N_PTS^3 = 16 million. Might be slow in Python.
        # Let's optimize: 
        # For a fixed pair (i, j), the points p that form an isosceles triangle with i, j are:
        # 1. On the perpendicular bisector of ij (dist(i,p) == dist(j,p))
        # 2. On the circle centered at i with radius dist(i,j) (dist(i,p) == dist(i,j))
        # 3. On the circle centered at j with radius dist(i,j) (dist(j,p) == dist(i,j))
        
        pi = ALL_POINTS[i]
        pj = ALL_POINTS[j]
        d2_ij = (pi[0]-pj[0])**2 + (pi[1]-pj[1])**2
        
        # Case 1: dist(i,p) == dist(j,p)
        # (p_x - i_x)^2 + (p_y - i_y)^2 = (p_x - j_x)^2 + (p_y - j_y)^2
        # This is a linear equation: 2(j_x-i_x)p_x + 2(j_y-i_y)p_y = j_x^2+j_y^2 - i_x^2-i_y^2
        dx = pj[0] - pi[0]
        dy = pj[1] - pi[1]
        rhs = pj[0]**2 + pj[1]**2 - pi[0]**2 - pi[1]**2
        # 2 dx p_x + 2 dy p_y = rhs
        # dx p_x + dy p_y = rhs / 2
        # Since p is integer, rhs must be even? Not necessarily, but 2 dx p_x + 2 dy p_y = rhs must hold.
        for k in range(N_PTS):
            if k == i or k == j:
                pk = ALL_POINTS[k]
                d2_ik = (pi[0]-pk[0])**2 + (pi[1]-pk[1])**2
                d2_jk = (pj[0]-pk[0])**2 + (pj[1]-pk[1])**2
                if d2_ik == d2_jk:
                    WITNESS_PAIRS[k].append((i, j))
        
        # Case 2: dist(i,p) == dist(i,j) => d2_ik == d2_ij
        for k in range(N_PTS):
            if k == i or k == j:
                pk = ALL_POINTS[k]
                d2_ik = (pi[0]-pk[0])**2 + (pi[1]-pk[1])**2
                if d2_ik == d2_ij:
                    WITNESS_PAIRS[k].append((i, j))
        
        # Case 3: dist(j,p) == dist(i,j) => d2_jk == d2_ij
        for k in range(N_PTS):
            if k == i or k == j:
                pk = ALL_POINTS[k]
                d2_jk = (pj[0]-pk[0])**2 + (pj[1]-pk[1])**2
                if d2_jk == d2_ij:
                    WITNESS_PAIRS[k].append((i, j))

# Deduplicate witness pairs for each k
for k in range(N_PTS):
    WITNESS_PAIRS[k] = list(set(WITNESS_PAIRS[k]))

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Initial solution: Random greedy
    def greedy_init():
        chosen = []
        available = list(range(N_PTS))
        random.shuffle(available)
        for idx in available:
            # Check if idx can be added
            can_add = True
            # Check isosceles with existing pairs
            for a in range(len(chosen)):
                for b in range(a+1, len(chosen)):
                    # Check if (idx, chosen[a], chosen[b]) is isosceles
                    p_idx = ALL_POINTS[idx]
                    p_a = ALL_POINTS[chosen[a]]
                    p_b = ALL_POINTS[chosen[b]]
                    d2_ab = (p_a[0]-p_b[0])**2 + (p_a[1]-p_b[1])**2
                    d2_ai = (p_a[0]-p_idx[0])**2 + (p_a[1]-p_idx[1])**2
                    d2_bi = (p_b[0]-p_idx[0])**2 + (p_b[1]-p_idx[1])**2
                    if d2_ab == d2_ai or d2_ab == d2_bi or d2_ai == d2_bi:
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                chosen.append(idx)
        return chosen
    
    best_solution = greedy_init()
    best_size = len(best_solution)
    
    # Simulated Annealing
    # State: list of selected indices
    # We will represent state as a set for O(1) lookups, but list for iteration.
    
    def is_valid_and_maximal(state_set):
        # Check isosceles constraint
        state_list = list(state_set)
        for i in range(len(state_list)):
            for j in range(i+1, len(state_list)):
                # Check if there is a k in state such that (i, j, k) is isosceles
                # Actually, we just need to check if ANY triple in state is isosceles.
                # We can precompute for each pair (i, j) the list of k that form isosceles.
                # But simpler: iterate all triples in state? No, O(|S|^3) is too slow.
                # We can check: for each pair (i, j) in state, if there is any k in state such that {i, j, k} is isosceles.
                # Precompute ISO_PAIR_TO_KS: dict mapping (i, j) to list of k.
                # We already have ISO_TRIPLES. Let's build a lookup.
                pass
        # This is getting complex. Let's use a simpler validity check for SA.
        # We will maintain the property that the state is always "iso-free" during SA moves.
        # Move: Add point p. Check if it creates an iso triangle with any pair in S.
        # Move: Remove point p. Always valid for iso constraint.
        # Move: Swap p (in S) with q (not in S). Check if q creates iso with S\{p}.
        
        # Maximal check:
        for p in range(N_PTS):
            if p in state_set:
                continue
            # Check if p is blocked
            blocked = False
            for i, j in WITNESS_PAIRS[p]:
                if i in state_set and j in state_set:
                    blocked = True
                    break
            if not blocked:
                return False
        return True

    # Precompute ISO_PAIR_TO_KS for fast validity check on add
    ISO_PAIR_TO_KS = {}
    for i, j, k in ISO_TRIPLES:
        if i > j:
            i, j = j, i
        if (i, j) not in ISO_PAIR_TO_KS:
            ISO_PAIR_TO_KS[(i, j)] = []
        ISO_PAIR_TO_KS[(i, j)].append(k)
        # Also store for (j, i)? No, we always use sorted pair.
        # When adding p, we check pairs (p, q) for q in S.
        # We need to know if there is any k in S such that {p, q, k} is iso.
        # This is equivalent to: for each q in S, is there k in S such that (p, q, k) is iso?
        # Or: for each q in S, check if p is in ISO_PAIR_TO_KS[(min(p,q), max(p,q))]? No, that's for fixed pair.
        # Let's precompute for each point p, the set of points q such that there exists k where {p, q, k} is iso?
        # No, we need to know if the SPECIFIC k is in S.
        
        # Efficient check for adding p to S:
        # For each q in S:
        #   For each k in ISO_TRIPLES containing (p, q):
        #      If k in S: Invalid.
        
        # Let's precompute: POINT_PAIR_TO_KS[p, q] = list of k such that {p, q, k} is isosceles.
        POINT_PAIR_TO_KS = {}
        for i, j, k in ISO_TRIPLES:
            # Add for all 3 pairs
            if (i, j) not in POINT_PAIR_TO_KS: POINT_PAIR_TO_KS[(i, j)] = []
            POINT_PAIR_TO_KS[(i, j)].append(k)
            if (j, i) not in POINT_PAIR_TO_KS: POINT_PAIR_TO_KS[(j, i)] = []
            POINT_PAIR_TO_KS[(j, i)].append(k)
            
            if (i, k) not in POINT_PAIR_TO_KS: POINT_PAIR_TO_KS[(i, k)] = []
            POINT_PAIR_TO_KS[(i, k)].append(j)
            if (k, i) not in POINT_PAIR_TO_KS: POINT_PAIR_TO_KS[(k, i)] = []
            POINT_PAIR_TO_KS[(k, i)].append(j)
            
            if (j, k) not in POINT_PAIR_TO_KS: POINT_PAIR_TO_KS[(j, k)] = []
            POINT_PAIR_TO_KS[(j, k)].append(i)
            if (k, j) not in POINT_PAIR_TO_KS: POINT_PAIR_TO_KS[(k, j)] = []
            POINT_PAIR_TO_KS[(k, j)].append(i)

    def can_add(p, state_set):
        for q in state_set:
            if (p, q) in POINT_PAIR_TO_KS:
                for k in POINT_PAIR_TO_KS[(p, q)]:
                    if k in state_set:
                        return False
        return True

    current_state = set(best_solution)
    current_size = len(current_state)
    
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    steps = 0
    max_steps = 100000
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Choose a random move
        move_type = random.random()
        if move_type < 0.3:
            # Add a point
            candidates = [p for p in range(N_PTS) if p not in current_state]
            if not candidates:
                continue
            p = random.choice(candidates)
            if can_add(p, current_state):
                new_state = current_state | {p}
                # Check maximality? No, SA allows temporary invalidity for maximality?
                # No, we must maintain iso-freeness. Maximality is the hard constraint for the final answer.
                # We can allow non-maximal states during SA to escape local minima, but we must penalize them.
                # Let's define cost = size + penalty * (number of unblocked points)
                # Unblocked points = points not in S that are not dominated.
                
                # For now, let's stick to valid states (iso-free) and only accept if maximal?
                # That's just local search.
                # Let's allow non-maximal states but penalize heavily.
                
                # Cost function:
                # size = len(state)
                # violations = number of points p not in state that are NOT blocked.
                # cost = size + 10 * violations
                
                # Calculate violations for new_state
                violations = 0
                for p_check in range(N_PTS):
                    if p_check in new_state:
                        continue
                    blocked = False
                    for i, j in WITNESS_PAIRS[p_check]:
                        if i in new_state and j in new_state:
                            blocked = True
                            break
                    if not blocked:
                        violations += 1
                        
                new_cost = len(new_state) + 10 * violations
                current_cost = current_size + 10 * 0 # Assume current is maximal?
                # We need to track current violations too.
                
                # Let's maintain current_violations
                pass
        else:
            # Remove a point
            if not current_state:
                continue
            p = random.choice(list(current_state))
            new_state = current_state - {p}
            # Removal is always iso-valid.
            # Check violations.
            violations = 0
            for p_check in range(N_PTS):
                if p_check in new_state:
                    continue
                blocked = False
                for i, j in WITNESS_PAIRS[p_check]:
                    if i in new_state and j in new_state:
                        blocked = True
                        break
                if not blocked:
                    violations += 1
            new_cost = len(new_state) + 10 * violations
            # Compare with current cost
            
        # This is getting messy with the cost function.
        # Let's simplify: Only consider moves that maintain iso-freeness.
        # Acceptance:
        # If new state is maximal: accept if size <= current size or with probability exp(-delta/T)
        # If new state is not maximal: accept with probability exp(-large_penalty/T)
        
        # To speed up, we will only evaluate maximality if the move is potentially beneficial.
        
        # Let's restart with a cleaner SA implementation.
        pass

    # Fallback: return best found so far
    return list(best_solution)

# The above logic is getting too complex for a quick rewrite.
# Let's use a simpler "Iterated Local Search" with a better perturbation.
# The previous program used greedy + removal.
# New strategy: 
# 1. Start with a random valid set.
# 2. Try to remove points (keep maximal).
# 3. If stuck, perturb by removing a random point and repairing maximality by adding minimal points.
# 4. Repeat.

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Precompute helpers
    def is_iso(a, b, c):
        ab2 = (a[0]-b[0])**2 + (a[1]-b[1])**2
        ac2 = (a[0]-c[0])**2 + (a[1]-c[1])**2
        bc2 = (b[0]-c[0])**2 + (b[1]-c[1])**2
        return ab2 == ac2 or ab2 == bc2 or ac2 == bc2

    def can_add(p, chosen_set):
        for q in chosen_set:
            for r in chosen_set:
                if q == r: continue
                if is_iso(p, q, r):
                    return False
        return True

    def is_maximal(chosen_set):
        for p in ALL_POINTS:
            if p in chosen_set:
                continue
            if can_add(p, chosen_set):
                return False
        return True

    def greedy_construct():
        chosen = []
        chosen_set = set()
        order = list(ALL_POINTS)
        random.shuffle(order)
        for p in order:
            if can_add(p, chosen_set):
                chosen.append(p)
                chosen_set.add(p)
        return chosen_set

    best_set = greedy_construct()
    best_size = len(best_set)
    
    # Local improvement: remove points
    def local_improve(s):
        improved = True
        while improved:
            improved = False
            for p in list(s):
                if time.time() > deadline:
                    return s
                test = s - {p}
                if is_maximal(test):
                    s = test
                    improved = True
        return s

    best_set = local_improve(best_set)
    best_size = len(best_set)
    
    # Iterated Local Search
    while time.time() < deadline - 10:
        # Perturb: remove k random points
        k = random.randint(1, 3)
        current = set(best_set)
        to_remove = random.sample(list(current), min(k, len(current)))
        for p in to_remove:
            current.remove(p)
        
        # Repair: Add points until maximal
        # This is tricky. If we remove points, we might lose maximality.
        # We need to add points to restore maximality, but also maintain iso-freeness.
        # Greedy repair:
        # While not maximal:
        #   Find a point p that can be added.
        #   Add p that minimizes future conflicts? Or just random.
        
        # Actually, removing points makes it EASIER to add points.
        # If current is not maximal, there exists p that can be added.
        # We add p. This might make it non-maximal again? No, adding points only blocks more points.
        # Wait. If S is not maximal, there is p that can be added.
        # If we add p, S' = S U {p}.
        # Is S' maximal? Not necessarily. There might be q that can be added to S' but not S?
        # No, if q can be added to S', it can be added to S (since S is subset of S').
        # So if S is not maximal, S' might be maximal or not.
        # Actually, adding a point can only REDUCE the set of addable points.
        # So if S is not maximal, S' might become maximal.
        
        # So the repair is:
        while not is_maximal(current):
            # Find a point that can be added
            addable = [p for p in ALL_POINTS if p not in current and can_add(p, current)]
            if not addable:
                break # Should not happen if is_maximal is correct
            # Choose one to add. 
            # To keep size small, we want to add points that block many others?
            # Or just random.
            p = random.choice(addable)
            current.add(p)
            
        # Local improvement
        current = local_improve(current)
        
        if len(current) < best_size:
            best_set = current
            best_size = len(current)
            
    return list(best_set)

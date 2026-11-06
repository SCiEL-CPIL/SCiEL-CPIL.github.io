import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 90s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    n = len(candidates)
    
    # 1. Partition numbers by their odd part
    # odd_part(x) = x / (2^v2(x))
    # If two numbers have the same odd part d, say x = d*2^a, y = d*2^b,
    # then lcm(x, y) = d * 2^max(a,b).
    # If d is small, say d <= 128, then d * 2^max(a,b) can be <= 256.
    # Specifically, if a,b >= 0, lcm <= 256 iff d * 2^max(a,b) <= 256.
    # This creates a "clique" or chain of conflicts within the same odd part group.
    
    # Let's group candidates by their odd part.
    groups = {}
    for x in candidates:
        d = x
        while d % 2 == 0:
            d //= 2
        if d not in groups:
            groups[d] = []
        groups[d].append(x)
    
    # For each group, we can select at most one number?
    # Let's check:
    # If x, y in group d, lcm(x,y) = d * 2^max(v2(x), v2(y)).
    # We need lcm(x,y) > 256.
    # If d * 2^max(v2(x), v2(y)) <= 256, they conflict.
    # Note that if we pick the largest number in the group, say x_max = d * 2^k,
    # then for any other y = d * 2^j with j < k, lcm(x_max, y) = x_max.
    # So x_max conflicts with y iff x_max <= 256.
    # Since all candidates are <= 256, x_max <= 256 is always true.
    # Therefore, ANY two numbers in the same odd-part group conflict?
    # Let's verify.
    # x = d * 2^a, y = d * 2^b. Assume a >= b.
    # lcm(x,y) = d * 2^a = x.
    # Condition: x <= 256.
    # Since x is in [2, 256], x <= 256 is always true.
    # Thus, lcm(x,y) <= 256 is ALWAYS true for distinct x,y in the same odd-part group.
    # Conclusion: We can select AT MOST ONE number from each odd-part group.
    
    # This is a HUGE simplification.
    # The problem reduces to:
    # 1. For each odd d, we have a set of candidates G_d = {d*2^k | d*2^k <= 256, k>=0}.
    # 2. We must choose at most one x from each G_d.
    # 3. If we choose x from G_d and y from G_e (d != e), we need lcm(x,y) > 256.
    # 4. Maximize sum 1/x.
    
    # Since we want to maximize sum 1/x, and for a fixed d, the smallest x in G_d is d (k=0),
    # and 1/x is decreasing, we should generally prefer smaller x.
    # However, choosing a smaller x (lower power of 2) might create MORE conflicts with other groups?
    # Let's analyze conflicts between different odd parts d and e.
    # x = d * 2^a, y = e * 2^b.
    # lcm(x,y) = d * e * 2^(a+b) / gcd(d,e) * 2^(min(a,b))? No.
    # lcm(d*2^a, e*2^b) = lcm(d,e) * 2^max(a,b) ? No, that's only if gcd(d,e)=1?
    # Actually, lcm(x,y) = x * y / gcd(x,y).
    # Let d, e be odd.
    # gcd(x,y) = gcd(d,e) * 2^min(a,b).
    # lcm(x,y) = (d * 2^a * e * 2^b) / (gcd(d,e) * 2^min(a,b))
    #           = (d * e / gcd(d,e)) * 2^(a+b-min(a,b))
    #           = lcm(d,e) * 2^max(a,b).
    
    # So the condition is: lcm(d,e) * 2^max(a,b) > 256.
    
    # Let L = lcm(d,e).
    # We need L * 2^max(a,b) > 256.
    # This implies 2^max(a,b) > 256 / L.
    
    # If L > 256, then L * 2^0 > 256, so ANY choice of a,b works.
    # If L <= 256, we need max(a,b) to be large enough.
    # Specifically, let K = ceil(log2(256/L + epsilon)).
    # If max(a,b) >= K, then L * 2^max(a,b) >= L * 2^K > 256.
    
    # This structure suggests we can model this as a graph problem on the ODD numbers (d).
    # Each odd d has a "chain" of choices: d, 2d, 4d, ...
    # Choosing a higher power of 2 (larger x) increases the weight cost (1/x smaller)
    # but increases the "conflict tolerance" (higher 2^max(a,b) makes it easier to satisfy lcm > 256 with neighbors).
    
    # Wait, higher x means smaller weight. We want to maximize weight.
    # So we prefer small x (small a).
    # But small a might violate the lcm constraint with neighbors.
    
    # Let's re-evaluate the "at most one per group" insight.
    # Yes, it's solid.
    # Number of odd numbers in [2, 256] is roughly 128.
    # The "core" graph has ~128 vertices (the odd numbers).
    # Each vertex d has a list of possible values: d, 2d, 4d, ...
    # Weight of choosing d*2^k is 1/(d*2^k).
    # Conflict between d (choosing 2^a) and e (choosing 2^b):
    # Compatible iff lcm(d,e) * 2^max(a,b) > 256.
    
    # This is still a complex combinatorial optimization, but on ~128 nodes with local choices.
    # However, note that if we pick the SMALLEST possible value for d (which is d itself, a=0),
    # we have the highest weight.
    # If we pick d, and it conflicts with e (where e is picked as e, b=0),
    # then lcm(d,e) <= 256.
    # To resolve this, we can increase a or b.
    # Increasing a changes weight from 1/d to 1/(2d). Loss = 1/(2d).
    # Increasing b changes weight from 1/e to 1/(2e). Loss = 1/(2e).
    
    # This looks like a "minimum weight vertex cover" or similar on a conflict graph of the base odd numbers,
    # where "covering" an edge (d,e) means increasing the power of 2 for d or e (or both) to break the conflict.
    
    # Let's build the base conflict graph on odd numbers.
    # Nodes: odd numbers in [3, 255] (1 is excluded by problem statement? "subset of [2,256]", 1 is not in range).
    # Actually, odd numbers in [2, 256] are 3, 5, ..., 255.
    # Edge (d,e) exists if lcm(d,e) <= 256.
    # If we select d and e with a=0, b=0, they conflict.
    # To fix, we must set a>=1 or b>=1 (or higher).
    
    # Is it sufficient to just set a=1 or b=1?
    # If lcm(d,e) * 2 > 256, i.e., lcm(d,e) > 128, then yes.
    # If lcm(d,e) <= 128, we might need a=2 or b=2, etc.
    
    # Strategy:
    # 1. Identify all odd numbers D in [3, 255].
    # 2. For each d in D, the possible choices are d, 2d, 4d, ... up to 256.
    # 3. We want to choose one value from each d (or none) to maximize sum 1/x, subject to pairwise lcm > 256.
    # 4. Note that if we don't choose any value for a d, we get 0 weight.
    # 5. Since weights are positive, we likely want to choose something for most d.
    
    # Given the time limit, a full search on 128 nodes is still hard if done naively.
    # But the graph of odd numbers with edges lcm(d,e)<=256 is likely sparse or has small cliques.
    # Let's check the density.
    # If d=3, e=5, lcm=15 <= 256. Conflict.
    # If d=3, e=127, lcm=381 > 256. No conflict.
    
    # Heuristic Approach:
    # Start with all a=0 (choose d).
    # Find all conflicts.
    # Resolve conflicts by increasing powers of 2.
    # This is a local search / greedy repair.
    
    # Let's implement a greedy construction with local search on the odd-part representation.
    
    # Step 1: List odd numbers
    odd_numbers = [d for d in range(3, 256, 2)]
    
    # For each odd d, max power of 2 such that d*2^k <= 256
    # max_k[d]
    
    # Initial state: choose d (k=0) for all d.
    # State: dict {d: k}
    # Score: sum 1/(d*2^k)
    # Validity: for all pairs d,e, lcm(d,e)*2^max(k_d, k_e) > 256
    
    # Let's precompute lcm(d,e) for all pairs of odds.
    # n_odds ~ 127. Pairs ~ 8000. Fast.
    
    # We will use a simple local search:
    # 1. Start with all k=0.
    # 2. While there are conflicts:
    #    Pick a conflicting pair (d,e).
    #    Increase k_d or k_e (whichever gives smaller penalty in weight, or random) to resolve.
    #    Check if this creates new conflicts?
    #    Actually, increasing k only REDUCES conflicts (makes lcm larger). It never creates new conflicts.
    #    Wait.
    #    Condition: lcm(d,e) * 2^max(k_d, k_e) > 256.
    #    If we increase k_d, max(k_d, k_e) increases or stays same.
    #    So the LHS increases.
    #    So if it was valid, it stays valid.
    #    If it was invalid, it might become valid.
    #    Therefore, the set of valid configurations is "downward closed" in terms of k? No, "upward closed".
    #    If a configuration is valid, any configuration with larger k's is also valid.
    #    We want to MINIMIZE the sum of penalties (loss of weight) from the base state (all k=0).
    #    Loss for d is sum_{i=0}^{k_d-1} 1/(d*2^i) - 1/(d*2^k_d)? No.
    #    Base weight: 1/d.
    #    New weight: 1/(d*2^k).
    #    Loss: 1/d - 1/(d*2^k) = (2^k - 1) / (d * 2^k).
    
    # So the problem is:
    # Minimize sum_{d} Loss(d, k_d)
    # Subject to: for all pairs (d,e), if lcm(d,e) <= 256, then 2^max(k_d, k_e) > 256 / lcm(d,e).
    # Let R(d,e) = ceil(log2(256 / lcm(d,e) + epsilon)).
    # Constraint: max(k_d, k_e) >= R(d,e).
    # Note: If lcm(d,e) > 256, R(d,e) = 0 (no constraint, since max(k)>=0 always).
    
    # This is a constraint satisfaction / optimization problem.
    # Variables: k_d >= 0 integers.
    # Constraints: max(k_d, k_e) >= R(d,e) for all edges in conflict graph G_conflict.
    # Objective: Minimize sum C_d(k_d).
    
    # Since C_d(k) is convex and increasing, and constraints are monotone,
    # a greedy approach might work well.
    # Or we can model this as a shortest path / min cut?
    # Actually, this looks like a "minimum weight vertex cover" variant if R(d,e) were 1.
    # But R(d,e) varies.
    
    # Simple Greedy:
    # 1. Start with all k=0.
    # 2. Identify all violated constraints (edges where max(k_d, k_e) < R(d,e)).
    # 3. For each violated edge, we must increase k_d or k_e to meet the requirement.
    # 4. Choose the increase that minimizes marginal cost.
    # 5. Repeat until no violations.
    
    # Will this be optimal? Not necessarily, but likely close.
    # We can also try local search: try to decrease k_d if it doesn't violate any constraint.
    
    # Let's implement this greedy + local search.
    
    # Precompute R(d,e) for all pairs.
    # R(d,e) = 0 if lcm(d,e) > 256.
    # Else, smallest k such that 2^k > 256/lcm(d,e).
    # i.e., 2^k * lcm(d,e) > 256.
    
    n_odds = len(odd_numbers)
    odd_to_idx = {d: i for i, d in enumerate(odd_numbers)}
    
    # Adjacency list for conflicts (where R > 0)
    # And required R values.
    # R_matrix[i][j]
    
    R_req = [[0]*n_odds for _ in range(n_odds)]
    for i in range(n_odds):
        d = odd_numbers[i]
        for j in range(i+1, n_odds):
            e = odd_numbers[j]
            l = lcm(d, e)
            if l <= 256:
                # Find min k such that l * 2^k > 256
                k = 0
                val = l
                while val <= 256:
                    val *= 2
                    k += 1
                R_req[i][j] = k
                R_req[j][i] = k
            # else R=0
    
    # Initialize k = 0 for all
    k = [0] * n_odds
    
    # Calculate current loss
    def calc_loss():
        total = 0.0
        for i, d in enumerate(odd_numbers):
            if k[i] > 0:
                # Loss = 1/d - 1/(d*2^k)
                total += (1.0/d) * (1.0 - 1.0/(2.0**k[i]))
        return total
    
    def is_valid():
        for i in range(n_odds):
            if k[i] == 0: continue # Optimization? No, need to check all
            for j in range(i+1, n_odds):
                if R_req[i][j] > 0:
                    if max(k[i], k[j]) < R_req[i][j]:
                        return False
        return True
    
    # Greedy Repair
    # While not valid:
    #   Find a violated pair (i,j)
    #   Need to increase k_i or k_j to at least R_req[i][j]
    #   Option A: set k_i = max(k_i, R_req[i][j])
    #   Option B: set k_j = max(k_j, R_req[i][j])
    #   Calculate cost delta for A and B.
    #   Choose cheaper.
    
    # Cost of increasing k_d from old_k to new_k:
    # Delta Loss = (1/d) * (1/(2^old_k) - 1/(2^new_k))
    
    def marginal_cost(d, old_k, new_k):
        if new_k <= old_k:
            return 0.0
        return (1.0/d) * (1.0/(2.0**old_k) - 1.0/(2.0**new_k))
    
    # We need to be careful: increasing k_i might satisfy multiple constraints.
    # A simple iterative repair might get stuck in local minima or be slow if we just pick one edge at a time.
    # Better: Iterate through all edges, if violated, fix it.
    # Or: Priority queue of violations?
    
    # Let's try a simple loop:
    # For a few iterations:
    #   changed = False
    #   for i in range(n_odds):
    #     for j in range(i+1, n_odds):
    #       if R_req[i][j] > 0:
    #         req = R_req[i][j]
    #         if k[i] < req and k[j] < req:
    #            # Both need to be increased? No, max(k_i, k_j) >= req.
    #            # So at least one must be >= req.
    #            cost_i = marginal_cost(odd_numbers[i], k[i], req)
    #            cost_j = marginal_cost(odd_numbers[j], k[j], req)
    #            if cost_i <= cost_j:
    #               k[i] = req
    #            else:
    #               k[j] = req
    #            changed = True
    #         elif k[i] < req:
    #            # k[j] is already >= req, so max is ok?
    #            # Wait. If k[j] >= req, then max(k_i, k_j) >= req. So it's valid.
    #            pass
    #         elif k[j] < req:
    #            pass
    #   if not changed: break
    
    # This loop might be slow if n_odds is 127 (8000 pairs) and many iterations.
    # But n_odds is small.
    
    # Let's run this repair.
    
    max_iters = 1000
    for _ in range(max_iters):
        changed = False
        for i in range(n_odds):
            d_i = odd_numbers[i]
            for j in range(i+1, n_odds):
                req = R_req[i][j]
                if req == 0:
                    continue
                if k[i] < req and k[j] < req:
                    d_j = odd_numbers[j]
                    cost_i = marginal_cost(d_i, k[i], req)
                    cost_j = marginal_cost(d_j, k[j], req)
                    if cost_i <= cost_j:
                        k[i] = req
                    else:
                        k[j] = req
                    changed = True
        if not changed:
            break
    
    # Now we have a valid configuration.
    # Can we improve?
    # Local Search: Try to decrease k_i.
    # If we decrease k_i, we might violate constraints with neighbors j where R_req[i][j] > k_i_new.
    # Specifically, for any neighbor j, we need max(k_i_new, k_j) >= R_req[i][j].
    # If k_j >= R_req[i][j], then k_i can be anything < R_req[i][j].
    # If k_j < R_req[i][j], then we need k_i >= R_req[i][j].
    # So min_k_i = max(0, max_{j: k_j < R_req[i][j]} R_req[i][j]).
    # If k_i > min_k_i, we can decrease k_i to min_k_i to save cost.
    
    # Repeat this improvement until no more improvements.
    
    for _ in range(100):
        improved = False
        for i in range(n_odds):
            min_k = 0
            for j in range(n_odds):
                if i == j: continue
                req = R_req[i][j]
                if req > 0:
                    if k[j] < req:
                        if min_k < req:
                            min_k = req
            if k[i] > min_k:
                k[i] = min_k
                improved = True
        if not improved:
            break
    
    # Construct result
    result = []
    for i, d in enumerate(odd_numbers):
        x = d * (2 ** k[i])
        if x <= 256:
            result.append(x)
    
    # Verify validity
    # for a in result:
    #   for b in result:
    #     if a != b and lcm(a,b) <= 256:
    #       return [2] # Fallback
    
    # Calculate exact sum for final check?
    # The task asks to return the subset. The evaluator calculates the score.
    # We just return the list.
    
    if not result:
        return [2]
        
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Result size: {len(res)}")
    print(f"Sum: {sum(1/x for x in res)}")
    # Print first few
    print(res[:20])

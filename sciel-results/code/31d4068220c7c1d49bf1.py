import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    N = 256

    # Precompute all 4-APs and 3-APs
    four_aps = []
    three_aps = []

    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            e = a + 3 * d
            if e <= N:
                four_aps.append((a, a + d, a + 2 * d, e))

    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            c = a + 2 * d
            if c <= N:
                three_aps.append((a, a + d, c))

    # For each element, precompute which 4-APs and 3-APs it participates in
    elem_in_4ap = {x: [] for x in range(1, N + 1)}
    for ap in four_aps:
        for x in ap:
            elem_in_4ap[x].append(ap)

    elem_in_3ap = {x: [] for x in range(1, N + 1)}
    for ap in three_aps:
        for x in ap:
            elem_in_3ap[x].append(ap)

    # Max-SAT formulation:
    # Hard clauses: For each 4-AP (a,b,c,e), NOT(all four in A)
    #   i.e., at least one of {a,b,c,e} must be excluded
    # Soft clauses: For each 3-AP (a,b,c), reward if all three in A
    #
    # WalkSAT-style: assign variables (in A or not), flip variables to
    # minimize hard violations + maximize soft satisfaction.
    #
    # Objective: score = (# satisfied soft clauses) - LARGE_PENALTY * (# violated hard clauses)

    LARGE_PENALTY = 10000  # Must exceed max possible 3-AP count

    def count_4ap_violations(A):
        """Count 4-APs fully contained in A."""
        count = 0
        for ap in four_aps:
            if all(x in A for x in ap):
                count += 1
        return count

    def count_3ap_satisfied(A):
        """Count 3-APs fully contained in A."""
        count = 0
        for ap in three_aps:
            if all(x in A for x in ap):
                count += 1
        return count

    def maxsat_score(A):
        """Max-SAT objective: satisfied soft - penalty * violated hard."""
        s = count_3ap_satisfied(A)
        h = count_4ap_violations(A)
        return s - LARGE_PENALTY * h

    def maxsat_walksat(A, time_limit):
        """
        WalkSAT-style local search on the Max-SAT formulation.
        
        Variables: x_i = 1 if i in A, 0 otherwise.
        Hard clauses: for each 4-AP, at least one variable is 0.
        Soft clauses: for each 3-AP, all three variables are 1.
        
        At each step, pick a variable to flip (add or remove from A).
        Score the flip by its effect on the Max-SAT objective.
        With some probability, pick a random variable (to escape local optima).
        """
        A = set(A)
        current_score = maxsat_score(A)
        best_A = set(A)
        best_score = current_score
        
        # Track clause weights (dynamic weighting like Weighted Max-SAT)
        # For hard clauses (4-APs): weight increases when violated
        # For soft clauses (3-APs): weight increases when not satisfied
        hard_weights = [1.0] * len(four_aps)
        soft_weights = [1.0] * len(three_aps)
        
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < time_limit:
            if no_improve > max_no_improve:
                # Reset weights and try harder exploration
                no_improve = 0
                max_no_improve = int(max_no_improve * 1.5)
                if max_no_improve > 5000:
                    break
                # Random restart: perturb current solution
                for _ in range(random.randint(3, 10)):
                    if A:
                        x = random.choice(list(A))
                        A.remove(x)
                    else:
                        break
                current_score = maxsat_score(A)
                continue
            
            # Choose a variable to flip
            flip_x = None
            
            # With probability 0.3, random flip (exploration)
            if random.random() < 0.3:
                if A:
                    flip_x = random.choice(list(A))
                    flip_to_remove = True
                else:
                    flip_x = random.randint(1, N)
                    flip_to_remove = False
            else:
                # Greedy: find the flip that most improves the score
                best_delta = -float('inf')
                best_flip = None
                best_flip_remove = None
                
                # Consider removing elements from A
                for x in A:
                    # If we remove x, which 3-APs are lost?
                    loss = 0
                    for ap in elem_in_3ap[x]:
                        others = [y for y in ap if y != x]
                        if all(y in A for y in others):
                            loss += 1
                    # If we remove x, which 4-AP violations are fixed?
                    gain_hard = 0
                    for ap in four_aps:
                        if x in ap:
                            others = [y for y in ap if y != x]
                            if all(y in A for y in others):
                                gain_hard += 1
                    delta = -loss + LARGE_PENALTY * gain_hard
                    if delta > best_delta:
                        best_delta = delta
                        best_flip = x
                        best_flip_remove = True
                
                # Consider adding elements not in A
                for x in range(1, N + 1):
                    if x in A:
                        continue
                    # If we add x, which 3-APs are gained?
                    gain = 0
                    for ap in elem_in_3ap[x]:
                        others = [y for y in ap if y != x]
                        if all(y in A for y in others):
                            gain += 1
                    # If we add x, which 4-AP violations are created?
                    loss_hard = 0
                    for ap in four_aps:
                        if x in ap:
                            others = [y for y in ap if y != x]
                            if all(y in A for y in others):
                                loss_hard += 1
                    delta = gain - LARGE_PENALTY * loss_hard
                    if delta > best_delta:
                        best_delta = delta
                        best_flip = x
                        best_flip_remove = False
                
                if best_flip is not None:
                    flip_x = best_flip
                    flip_to_remove = best_flip_remove
            
            if flip_x is None:
                break
            
            # Perform the flip
            if flip_to_remove:
                A.remove(flip_x)
            else:
                A.add(flip_x)
            
            new_score = maxsat_score(A)
            
            if new_score > current_score:
                current_score = new_score
                no_improve = 0
                if new_score > best_score:
                    best_score = new_score
                    best_A = set(A)
                # Update weights: increase weights of satisfied soft clauses
                # and decrease weights of violated hard clauses (or vice versa)
                # Actually, standard weighted Max-SAT: increase weight of unsatisfied
                # clauses to make them more "important"
            else:
                no_improve += 1
                # Update clause weights (Weighted Max-SAT style)
                # Increase weight of violated hard clauses
                for i, ap in enumerate(four_aps):
                    if all(x in A for x in ap):
                        hard_weights[i] *= 1.1
                # Increase weight of unsatisfied soft clauses (those with exactly 2 in A)
                for i, ap in enumerate(three_aps):
                    in_count = sum(1 for x in ap if x in A)
                    if in_count == 2:
                        soft_weights[i] *= 1.05
                # Normalize weights occasionally
                if no_improve % 50 == 0:
                    max_hw = max(hard_weights) if hard_weights else 1.0
                    max_sw = max(soft_weights) if soft_weights else 1.0
                    if max_hw > 100:
                        hard_weights = [w / max_hw for w in hard_weights]
                    if max_sw > 100:
                        soft_weights = [w / max_sw for w in soft_weights]
        
        return best_A, best_score

    def greedy_construct():
        """Greedy construction: add elements that maximize 3-AP gain without creating 4-APs."""
        A = set()
        remaining = set(range(1, N + 1))
        
        while remaining:
            best_x = None
            best_gain = -1
            for x in remaining:
                # Check if adding x creates a 4-AP
                creates_4ap = False
                for ap in elem_in_4ap[x]:
                    others = [y for y in ap if y != x]
                    if all(y in A for y in others):
                        creates_4ap = True
                        break
                if creates_4ap:
                    continue
                # Count 3-AP gain
                gain = 0
                for ap in elem_in_3ap[x]:
                    others = [y for y in ap if y != x]
                    if all(y in A for y in others):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            A.add(best_x)
            remaining.remove(best_x)
        
        return A, remaining

    # Multi-restart strategy
    best_A = set()
    best_score = -1

    num_restarts = 12
    per_restart_time = (deadline - start_time - 5.0) / num_restarts

    for restart in range(num_restarts):
        restart_deadline = start_time + 5.0 + (restart + 1) * per_restart_time
        if time.time() > restart_deadline - 1:
            break

        # Greedy construction
        A, remaining = greedy_construct()
        current_score = maxsat_score(A)
        
        if current_score > best_score:
            best_score = current_score
            best_A = set(A)

        # WalkSAT local search
        if time.time() < restart_deadline - 1:
            ls_A, ls_score = maxsat_walksat(A, restart_deadline - 1)
            if ls_score > best_score:
                best_score = ls_score
                best_A = ls_A

    # Final polish
    if time.time() < deadline - 2 and best_A:
        final_A, final_score = maxsat_walksat(best_A, deadline - 2)
        if final_score > best_score:
            best_score = final_score
            best_A = final_A

    # Ensure no 4-APs (safety check)
    # Remove elements to break any remaining 4-APs
    changed = True
    while changed:
        changed = False
        for ap in four_aps:
            if all(x in best_A for x in ap):
                # Remove the element that loses the fewest 3-APs
                min_loss = float('inf')
                min_x = None
                for x in ap:
                    loss = 0
                    for tap in elem_in_3ap[x]:
                        others = [y for y in tap if y != x]
                        if all(y in best_A for y in others):
                            loss += 1
                    if loss < min_loss:
                        min_loss = loss
                        min_x = x
                best_A.remove(min_x)
                changed = True
                break

    return sorted(best_A)

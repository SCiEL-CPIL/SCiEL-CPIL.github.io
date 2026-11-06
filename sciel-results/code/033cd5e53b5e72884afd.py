import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    base = [1, 2, 5, 11]
    base_set = set(base)
    all_numbers = list(range(1, 285))
    
    best_size = len(base)
    best_solution = base[:]

    def compute_sums(lst):
        s = set()
        n = len(lst)
        for i in range(n):
            a = lst[i]
            for j in range(i, n):
                s.add(a + lst[j])
        return s

    def try_add(lst, sums, x):
        """Check if x can be added. Returns (ok, new_sums) or (False, None)."""
        # Check x+y for y in lst
        for y in lst:
            if x + y in sums:
                return False, None
        if 2 * x in sums:
            return False, None
        
        new_sums = sums | set()
        for y in lst:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        return True, new_sums

    def try_remove(lst, sums, x):
        """Remove x from lst and sums. Returns (new_lst, new_sums)."""
        new_sums = sums | set()
        for y in lst:
            if y != x:
                new_sums.discard(x + y)
        new_sums.discard(2 * x)
        
        new_lst = [y for y in lst if y != x]
        return new_lst, new_sums

    def greedy_build():
        """Randomized greedy construction to get a starting point."""
        cur = base[:]
        sums = compute_sums(cur)
        used = set(cur)
        
        candidates = [x for x in all_numbers if x not in used]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 5.0:
                break
            ok, new_sums = try_add(cur, sums, x)
            if ok:
                cur.append(x)
                sums = new_sums
                used.add(x)
        
        return cur, sums

    # Simulated Annealing
    def simulated_annealing(init_list, init_sums, max_steps=5000, t_start=1.0, t_end=0.01):
        cur = init_list[:]
        cur_sums = init_sums | set()
        cur_size = len(cur)
        
        best_local_size = cur_size
        best_local_solution = cur[:]
        
        # Temperature schedule
        # We want to run for a fixed duration or steps, cooling down
        # Let's use steps-based cooling
        step = 0
        t = t_start
        
        # Precompute non-base elements for removal candidates
        # We need to be able to remove non-base elements
        
        while step < max_steps and time.time() < deadline - 5.0:
            step += 1
            t = t_start * (t_end / t_start) ** (step / max_steps)
            
            # Choose a move
            # 1. Try to add a random unused number
            # 2. Try to remove a random non-base number
            # 3. Try to swap (remove one, add one)
            
            move_type = random.random()
            
            new_cur = cur
            new_sums = cur_sums
            new_size = cur_size
            
            # Determine candidates
            used_set = set(cur)
            
            if move_type < 0.4:
                # Try Add
                unused = [x for x in all_numbers if x not in used_set]
                if unused:
                    x = random.choice(unused)
                    ok, new_sums_candidate = try_add(cur, cur_sums, x)
                    if ok:
                        new_cur = cur + [x]
                        new_sums = new_sums_candidate
                        new_size = cur_size + 1
            
            elif move_type < 0.7:
                # Try Remove
                non_base = [x for x in cur if x not in base_set]
                if non_base:
                    x = random.choice(non_base)
                    new_cur, new_sums = try_remove(cur, cur_sums, x)
                    new_size = cur_size - 1
            else:
                # Try Swap: Remove one non-base, add one unused
                non_base = [x for x in cur if x not in base_set]
                unused = [x for x in all_numbers if x not in used_set]
                if non_base and unused:
                    x_remove = random.choice(non_base)
                    x_add = random.choice(unused)
                    
                    # First remove
                    temp_cur, temp_sums = try_remove(cur, cur_sums, x_remove)
                    # Then add
                    ok, final_sums = try_add(temp_cur, temp_sums, x_add)
                    if ok:
                        new_cur = temp_cur + [x_add]
                        new_sums = final_sums
                        new_size = cur_size
                    else:
                        # Move invalid, skip
                        continue
            
            # Calculate delta
            delta = new_size - cur_size
            
            # Acceptance criterion
            if delta > 0:
                # Always accept improvement
                cur = new_cur
                cur_sums = new_sums
                cur_size = new_size
                if cur_size > best_local_size:
                    best_local_size = cur_size
                    best_local_solution = cur[:]
            else:
                # Accept with probability exp(delta / t)
                # delta is negative or 0
                if delta == 0:
                    # Accept neutral moves with some probability to explore
                    if random.random() < 0.5:
                        cur = new_cur
                        cur_sums = new_sums
                else:
                    # delta < 0
                    prob = math.exp(delta / t)
                    if random.random() < prob:
                        cur = new_cur
                        cur_sums = new_sums
                        cur_size = new_size
            
        return best_local_solution, best_local_size

    # Main loop: Multiple restarts with SA
    # 1. Initial greedy
    # 2. SA on it
    # 3. If time remains, restart with a new greedy or kick the best
    
    # Phase 1: Initial Greedy + SA
    cur_list, cur_sums = greedy_build()
    sa_sol, sa_size = simulated_annealing(cur_list, cur_sums)
    if sa_size > best_size:
        best_size = sa_size
        best_solution = sa_sol
    
    # Phase 2: Restarting
    # We will perform several restarts, each doing a greedy build (with different randomness) 
    # and then a short SA run.
    
    # To make restarts more effective, we can "kick" the best solution found so far
    # and run SA on the kicked version.
    
    while time.time() < deadline - 10.0:
        # Decide: Fresh Greedy or Kick Best?
        if random.random() < 0.5:
            # Fresh greedy
            cur_list, cur_sums = greedy_build()
        else:
            # Kick best solution
            kick_size = random.choice([2, 3])
            non_base_in_best = [x for x in best_solution if x not in base_set]
            if len(non_base_in_best) < kick_size:
                continue
            to_remove = set(random.sample(non_base_in_best, kick_size))
            kicked_list = [x for x in best_solution if x not in to_remove]
            cur_sums = compute_sums(kicked_list)
            cur_list = kicked_list
        
        # Run SA with a time budget
        # Estimate steps based on remaining time? 
        # Or just fixed steps, checking time in loop.
        sa_sol, sa_size = simulated_annealing(cur_list, cur_sums, max_steps=2000)
        
        if sa_size > best_size:
            best_size = sa_size
            best_solution = sa_sol

    return best_solution

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)

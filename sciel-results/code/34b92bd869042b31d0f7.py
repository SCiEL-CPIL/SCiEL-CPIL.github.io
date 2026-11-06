import time
import random
import math

def construct():
    """
    Returns a list of 32 distinct integers in 1..512 maximizing the number
    of distinct differences d > 0 such that a, a+d, a+2d are in the set.
    """
    
    def score_set(S_list, S_set):
        valid_d = set()
        n = len(S_list)
        # Local variables for speed
        s_set = S_set
        s_list = S_list
        for i in range(n):
            a = s_list[i]
            for j in range(i + 1, n):
                b = s_list[j]
                d = b - a
                if b + d in s_set:
                    valid_d.add(d)
        return len(valid_d)

    def random_set():
        return random.sample(range(1, 513), 32)

    def geometric_set():
        # Log-spaced
        pts = set()
        for i in range(32):
            # Map i/31 to 0..1, then to 1..512 log scale
            # log(1) = 0, log(512) ~ 6.2
            frac = i / 31.0
            val = int(math.exp(frac * math.log(512)))
            if 1 <= val <= 512:
                pts.add(val)
        # Fill remaining
        while len(pts) < 32:
            pts.add(random.randint(1, 512))
        return list(pts)

    def greedy_construct():
        S_set = set()
        S_list = []
        for _ in range(32):
            best_x = -1
            best_score = -1
            # To make it diverse, we can shuffle the range or just iterate
            for x in range(1, 513):
                if x in S_set:
                    continue
                # Create new set
                new_list = S_list + [x]
                new_set = S_set | {x}
                s = score_set(new_list, new_set)
                if s > best_score:
                    best_score = s
                    best_x = x
            S_list.append(best_x)
            S_set.add(best_x)
        return S_list

    def simulated_annealing(start_list, T0, Tend, steps):
        S_list = list(start_list)
        S_set = set(S_list)
        current_score = score_set(S_list, S_set)
        best_list = list(S_list)
        best_score = current_score
        
        for step in range(steps):
            i = random.randint(0, 31)
            old_val = S_list[i]
            S_set.remove(old_val)
            
            new_val = random.randint(1, 512)
            if new_val in S_set:
                S_set.add(old_val)
                continue
            
            S_list[i] = new_val
            S_set.add(new_val)
            
            new_score = score_set(S_list, S_set)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_list = list(S_list)
            else:
                delta = new_score - current_score
                if delta < 0:
                    # exp(delta / T)
                    # delta is negative integer
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        S_set.remove(new_val)
                        S_set.add(old_val)
                        S_list[i] = old_val
                else:
                    current_score = new_score
            
            T = T0 * (Tend / T0) ** (step / steps)
            if T < Tend:
                T = Tend

        return best_list, best_score

    start_time = time.time()
    deadline = start_time + 140
    
    # Initial Candidates
    candidates = [
        list(range(1, 33)),
        random_set(),
        random_set(),
        geometric_set(),
        greedy_construct() # This takes ~1-2s
    ]
    
    # Evaluate
    scored = []
    for c in candidates:
        s = score_set(c, set(c))
        scored.append((s, c))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    best_global_score, best_global_list = scored[0]
    
    T0 = 5.0
    Tend = 0.1
    steps_per_sa = 1000
    
    # Search Loop
    while time.time() < deadline - 5:
        # 30% exploration, 70% exploitation
        if random.random() < 0.3:
            start_list = random_set()
        else:
            start_list = best_global_list
            
        res_list, res_score = simulated_annealing(start_list, T0, Tend, steps_per_sa)
        
        if res_score > best_global_score:
            best_global_score = res_score
            best_global_list = res_list

    # Final Polish
    remaining = deadline - time.time()
    if remaining > 2:
        # Run a few more SA runs on the best with lower T
        for _ in range(5):
            if time.time() > deadline - 1:
                break
            res_list, res_score = simulated_annealing(best_global_list, 1.0, 0.01, 2000)
            if res_score > best_global_score:
                best_global_score = res_score
                best_global_list = res_list

    return best_global_list

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)

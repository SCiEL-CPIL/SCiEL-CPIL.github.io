import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    def score_set(s):
        """Count distinct d > 0 such that a, a+d, a+2d all in s for some a."""
        s_set = set(s)
        distinct_d = set()
        for a in s:
            for b in s:
                if b == a:
                    continue
                d = b - a
                if d <= 0:
                    continue
                if a + 2 * d in s_set:
                    distinct_d.add(d)
        return len(distinct_d)
    
    def greedy_polish(s, time_limit):
        """Improve set s via greedy local search until time_limit."""
        s = list(s)
        current_score = score_set(s)
        while time.time() < time_limit:
            improved = False
            # Try moving each element to a new random position
            for i in range(len(s)):
                if time.time() >= time_limit:
                    break
                old_val = s[i]
                # Try several new values
                for _ in range(20):
                    new_val = random.randint(1, 512)
                    if new_val in s_set(s):
                        continue
                    s[i] = new_val
                    new_score = score_set(s)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        break
                    s[i] = old_val
            if not improved:
                break
        return s, current_score
    
    def s_set(s):
        return set(s)
    
    def simulated_annealing(s, time_limit):
        """SA on set s."""
        s = list(s)
        current_score = score_set(s)
        best_s = list(s)
        best_score = current_score
        temp = 100.0
        cooling = 0.9995
        
        while time.time() < time_limit and temp > 0.1:
            # Generate a move
            move_type = random.random()
            if move_type < 0.5:
                # Replace one element
                i = random.randint(0, len(s) - 1)
                old_val = s[i]
                new_val = random.randint(1, 512)
                if new_val == old_val or new_val in s_set(s) - {old_val}:
                    continue
                s[i] = new_val
                new_score = score_set(s)
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.718281828, delta / temp):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_s = list(s)
                else:
                    s[i] = old_val
            elif move_type < 0.8:
                # Swap two elements
                i, j = random.sample(range(len(s)), 2)
                s[i], s[j] = s[j], s[i]
                new_score = score_set(s)
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.718281828, delta / temp):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_s = list(s)
                else:
                    s[i], s[j] = s[j], s[i]
            else:
                # Move one element
                i = random.randint(0, len(s) - 1)
                old_val = s[i]
                new_val = random.randint(1, 512)
                if new_val == old_val or new_val in s_set(s) - {old_val}:
                    continue
                s[i] = new_val
                new_score = score_set(s)
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.718281828, delta / temp):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_s = list(s)
                else:
                    s[i] = old_val
            
            temp *= cooling
        
        return best_s, best_score
    
    # Generate initial candidates
    def random_initial():
        return random.sample(range(1, 513), 32)
    
    def structured_initial():
        # Try a few structured approaches
        methods = [
            lambda: list(range(1, 33)),  # consecutive
            lambda: [i * 16 for i in range(1, 33)],  # multiples of 16
            lambda: [1 + i * 16 for i in range(32)],  # 1, 17, 33, ...
            lambda: [i * i + 1 for i in range(1, 33)],  # squares + 1
            lambda: [1 << i for i in range(32)] if all(1 << i <= 512 for i in range(32)) else random_initial(),
        ]
        # Filter valid ones
        valid = []
        for m in methods:
            try:
                s = m()
                if len(set(s)) == 32 and all(1 <= x <= 512 for x in s):
                    valid.append(s)
            except:
                pass
        if valid:
            return random.choice(valid)
        return random_initial()
    
    # Multi-restart SA with greedy polish
    best_overall = None
    best_overall_score = 0
    
    num_restarts = 0
    while time.time() < deadline - 5:
        # Generate initial set
        if num_restarts < 4:
            s_init = structured_initial()
        else:
            s_init = random_initial()
        
        # Run SA
        sa_time_limit = time.time() + 20  # 20 seconds per SA run
        s_sa, score_sa = simulated_annealing(s_init, sa_time_limit)
        
        # Greedy polish
        polish_time_limit = min(time.time() + 10, deadline - 3)
        s_polished, score_polished = greedy_polish(s_sa, polish_time_limit)
        
        if score_polished > best_overall_score:
            best_overall = s_polished
            best_overall_score = score_polished
        
        num_restarts += 1
        
        # If we've found a good score, we can be more aggressive with restarts
        if best_overall_score >= 40:
            break
    
    # Final check
    if best_overall is None:
        best_overall = list(range(1, 33))
    
    # Ensure validity
    result = sorted(best_overall)
    if len(set(result)) != 32 or not all(1 <= x <= 512 for x in result):
        result = list(range(1, 33))
    
    return result

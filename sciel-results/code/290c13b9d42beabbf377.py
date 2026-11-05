import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # 90s limit, leave 5s margin
    
    def greedy_init(order):
        selected = []
        reachable = 1
        for x in order:
            new_reach = reachable | (reachable << x)
            if not (new_reach >> 200) & 1:
                selected.append(x)
                reachable = new_reach
        return set(selected)
    
    # Try multiple greedy strategies and keep the best
    best_init = set()
    best_init_size = 0
    
    s1 = greedy_init(range(100, 0, -1))
    if len(s1) > best_init_size:
        best_init = s1
        best_init_size = len(s1)
        
    s2 = greedy_init(range(1, 101))
    if len(s2) > best_init_size:
        best_init = s2
        best_init_size = len(s2)
        
    for _ in range(20):
        lst = list(range(1, 101))
        random.shuffle(lst)
        s3 = greedy_init(lst)
        if len(s3) > best_init_size:
            best_init = s3
            best_init_size = len(s3)
            
    current_set = best_init.copy()
    current_size = best_init_size
    
    best_set = current_set.copy()
    best_size = current_size
    
    all_elements = list(range(1, 101))
    
    def check_validity(s):
        r = 1
        for x in s:
            r |= (r << x)
            if (r >> 200) & 1:
                return False
        return True
    
    # Simulated annealing with directed improvements
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Phase 1: Directed improvement - remove 1, add 2 (net +1)
        if current_size > 0 and len(all_elements) - current_size >= 2:
            remove_x = random.choice(list(current_set))
            candidates = [x for x in all_elements if x not in current_set]
            
            # Try a few random pairs to add
            found_improvement = False
            for _ in range(30):
                if time.time() > deadline:
                    break
                if len(candidates) < 2:
                    break
                add_pair = random.sample(candidates, 2)
                new_set = current_set - {remove_x}
                new_set.update(add_pair)
                
                if check_validity(new_set) and len(new_set) > current_size:
                    current_set = new_set
                    current_size = len(new_set)
                    if current_size > best_size:
                        best_set = current_set.copy()
                        best_size = current_size
                    found_improvement = True
                    break
            
            if found_improvement:
                continue
        
        # Phase 2: Simple swap or flip for plateau exploration
        num_flips = random.randint(1, 2)
        new_set = current_set.copy()
        
        for _ in range(num_flips):
            x = random.choice(all_elements)
            if x in new_set:
                new_set.remove(x)
            else:
                new_set.add(x)
                
        if check_validity(new_set):
            new_size = len(new_set)
            if new_size > current_size:
                current_set = new_set
                current_size = new_size
                if new_size > best_size:
                    best_set = current_set.copy()
                    best_size = new_size
            elif new_size == current_size:
                if random.random() < 0.5:
                    current_set = new_set
            else:
                delta = current_size - new_size
                if random.random() < pow(2.71828, -delta / T):
                    current_set = new_set
                    current_size = new_size
        
        T *= cooling
        if T < 0.01:
            T = 1.0
            if random.random() < 0.1:
                current_set = best_set.copy()
                current_size = best_size

    return list(best_set)

import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    n = 512
    
    def check_adds_4ap(s, x):
        """Check if adding x to set s creates a nonconstant 4-term AP."""
        for d in range(1, n + 1):
            # x is first term
            if x + d in s and x + 2*d in s and x + 3*d in s:
                return True
            # x is second term
            if x - d in s and x + d in s and x + 2*d in s:
                return True
            # x is third term
            if x - 2*d in s and x - d in s and x + d in s:
                return True
            # x is fourth term
            if x - 3*d in s and x - 2*d in s and x - d in s:
                return True
        return False
    
    def greedy_forward():
        s = set()
        for x in range(1, n + 1):
            if not check_adds_4ap(s, x):
                s.add(x)
        return s
    
    def greedy_backward():
        s = set()
        for x in range(n, 0, -1):
            if not check_adds_4ap(s, x):
                s.add(x)
        return s
    
    def greedy_random(seed):
        rng = random.Random(seed)
        order = list(range(1, n + 1))
        rng.shuffle(order)
        s = set()
        for x in order:
            if not check_adds_4ap(s, x):
                s.add(x)
        return s

    # Start with forward greedy
    best_set = greedy_forward()
    best_size = len(best_set)
    
    # Try backward greedy
    if time.time() < deadline - 5:
        s = greedy_backward()
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Try several random greedy restarts
    seed = 42
    while time.time() < deadline - 5:
        s = greedy_random(seed)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
        seed += 1
        
    return sorted(best_set)

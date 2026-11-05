import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    def evaluate(seq):
        n = len(seq)
        if n == 0:
            return 0, True
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + seq[i]
        
        sums = set()
        dups = 0
        for i in range(n):
            pi = prefix[i]
            for j in range(i + 1, n + 1):
                s = prefix[j] - pi
                if s in sums:
                    dups += 1
                else:
                    sums.add(s)
        
        # Score: Heavily weight length, penalize duplicates
        score = n * 10000 - dups
        valid = (dups == 0)
        return score, valid

    def get_best_greedy():
        best_seq = []
        best_len = 0
        
        # Try multiple greedy strategies
        strategies = [
            lambda x: x,             # Smallest first
            lambda x: 256 - x + 1,   # Largest first
            lambda x: random.randint(1, 256) # Random order
        ]
        
        for _ in range(20): # Try 20 random orders and 2 fixed
            if _ < 2:
                order = list(range(1, 257))
                if _ == 0:
                    pass # Smallest
                else:
                    order.reverse() # Largest
            else:
                order = list(range(1, 257))
                random.shuffle(order)
            
            seq = []
            used = set()
            for x in order:
                if x in used: continue
                test_seq = seq + [x]
                test_seq.sort()
                score, valid = evaluate(test_seq)
                if valid:
                    seq = test_seq
                    used.add(x)
            
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
                
        return best_seq

    # 1. Initialize
    current_seq = get_best_greedy()
    current_score, current_valid = evaluate(current_seq)
    best_seq = current_seq[:]
    best_score = current_score

    all_nums = set(range(1, 257))
    
    # 2. Simulated Annealing
    T = 500.0
    cooling_rate = 0.995
    
    while time.time() < deadline:
        # Adaptive move selection
        if current_valid:
            move_type = random.choices(['add', 'remove', 'replace'], weights=[70, 10, 20])[0]
        else:
            move_type = random.choices(['add', 'remove', 'replace'], weights=[20, 30, 50])[0]
        
        new_seq = current_seq[:]
        
        if move_type == 'add':
            unused = [x for x in all_nums if x not in current_seq]
            if unused:
                x = random.choice(unused)
                new_seq.append(x)
                new_seq.sort()
        
        elif move_type == 'remove':
            if current_seq:
                x = random.choice(current_seq)
                new_seq.remove(x)
        
        elif move_type == 'replace':
            if current_seq:
                unused = [x for x in all_nums if x not in current_seq]
                if unused:
                    old_x = random.choice(current_seq)
                    new_x = random.choice(unused)
                    new_seq.remove(old_x)
                    new_seq.append(new_x)
                    new_seq.sort()
        
        if new_seq == current_seq:
            continue
        
        new_score, new_valid = evaluate(new_seq)
        
        delta = new_score - current_score
        
        # Acceptance probability
        if delta > 0:
            accept = True
        else:
            # Avoid overflow in exp
            if T > 0:
                accept = random.random() < math.exp(delta / T)
            else:
                accept = False
        
        if accept:
            current_seq = new_seq
            current_score = new_score
            current_valid = new_valid
            
            if new_valid and new_score > best_score:
                best_score = new_score
                best_seq = new_seq[:]
        
        T *= cooling_rate
        if T < 1.0:
            T = 1.0
            
    return best_seq

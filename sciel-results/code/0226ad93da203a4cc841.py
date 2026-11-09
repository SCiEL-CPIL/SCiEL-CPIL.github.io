# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    smooth_set = set(smooth)
    
    def is_antichain(terms):
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                a, b = terms[i], terms[j]
                if a > b:
                    a, b = b, a
                if b % a == 0:
                    return False
        return True
    
    answers = []
    for n in range(1, 129):
        if time.time() > DEADLINE:
            while len(answers) < 128:
                answers.append([1])
            return answers
        
        answer = None
        
        # Count 1
        if n in smooth_set:
            answer = [n]
        else:
            # Count 2
            for a in smooth:
                if a >= n:
                    break
                b = n - a
                if b in smooth_set and b > a:
                    if b % a != 0:
                        answer = [a, b]
                        break
            if answer is None:
                # Count 3
                for i in range(len(smooth)):
                    a = smooth[i]
                    if a >= n:
                        break
                    for j in range(i + 1, len(smooth)):
                        b = smooth[j]
                        if a + b >= n:
                            break
                        c = n - a - b
                        if c in smooth_set and c > b:
                            if is_antichain((a, b, c)):
                                answer = [a, b, c]
                                break
                    if answer is not None:
                        break
            if answer is None:
                # Count 4+
                for count in range(4, 31):
                    if time.time() > DEADLINE:
                        break
                    candidates = [x for x in smooth if x <= n]
                    if len(candidates) < count:
                        break
                    for terms in combinations(candidates, count):
                        if sum(terms) == n and is_antichain(terms):
                            answer = list(terms)
                            break
                    if answer is not None:
                        break
        
        answers.append(answer)
    
    return answers
# EVOLVE-BLOCK-END

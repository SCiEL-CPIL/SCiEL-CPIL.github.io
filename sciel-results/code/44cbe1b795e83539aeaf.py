import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin

    def score(S):
        S_set = set(S)
        count = 0
        for d in range(1, 256):
            found = False
            for a in S_set:
                if a + d in S_set and a + 2*d in S_set:
                    found = True
                    break
            if found:
                count += 1
        return count

    best_S = list(range(1, 33))
    best_score = score(best_S)

    # Multiple random restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break

        # Initialize with random set
        S = set()
        while len(S) < 32:
            S.add(random.randint(1, 512))
        
        current_score = score(S)
        
        # Hill climb
        stuck_count = 0
        while time.time() < deadline - 5:
            # Random move: replace one element
            x = random.choice(list(S))
            y = random.randint(1, 512)
            if y in S:
                continue
                
            new_S = set(S)
            new_S.remove(x)
            new_S.add(y)
            
            new_score = score(new_S)
            
            if new_score > current_score:
                S = new_S
                current_score = new_score
                stuck_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_S = list(S)
            else:
                stuck_count += 1
                # If stuck for a while, break to try another restart
                if stuck_count > 200:
                    break

    return best_S

import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    # The task is specifically K_3(6,1).
    # If parameters are different, we fall back to a simple greedy to ensure validity.
    if q != 3 or n != 6 or radius != 1:
        # Simple greedy for generic case (slow but valid)
        N = q ** n
        def int_to_word(x):
            w = [0] * n
            for i in range(n):
                w[i] = x % q
                x //= q
            return w
        def word_to_int(w):
            x = 0
            for i in range(n - 1, -1, -1):
                x = x * q + w[i]
            return x
        
        # Precompute balls for generic case
        balls = []
        for x in range(N):
            w = int_to_word(x)
            neighbors = [x]
            # Generate all words within distance radius
            # This is expensive for large n, but for small n it's ok.
            # For radius 1:
            if radius == 1:
                for i in range(n):
                    orig = w[i]
                    for d in range(1, q):
                        w[i] = (orig + d) % q
                        neighbors.append(word_to_int(w))
                    w[i] = orig
            else:
                # Generic radius handling is complex, skip for this specific task optimization
                # Just return a trivial code if not 3,6,1
                return [int_to_word(i) for i in range(N)]
            balls.append(neighbors)
        
        # Greedy
        uncovered = np.ones(N, dtype=bool)
        C = set()
        while np.any(uncovered):
            best_x = -1
            best_count = -1
            for x in range(N):
                cnt = sum(1 for nb in balls[x] if uncovered[nb])
                if cnt > best_count:
                    best_count = cnt
                    best_x = x
            C.add(best_x)
            for nb in balls[best_x]:
                uncovered[nb] = False
        return [int_to_word(x) for x in C]

    # Specific optimization for q=3, n=6, r=1
    N = 729
    def int_to_word(x):
        return [x % 3, (x // 3) % 3, (x // 9) % 3, (x // 27) % 3, (x // 81) % 3, (x // 243) % 3]
    
    def word_to_int(w):
        return w[0] + w[1]*3 + w[2]*9 + w[3]*27 + w[4]*81 + w[5]*243

    # Precompute balls
    balls = []
    for x in range(N):
        w = int_to_word(x)
        neighbors = [x]
        for i in range(6):
            orig = w[i]
            w[i] = (orig + 1) % 3
            neighbors.append(word_to_int(w))
            w[i] = (orig + 2) % 3
            neighbors.append(word_to_int(w))
            w[i] = orig
        balls.append(neighbors)

    def greedy_init():
        uncovered = np.ones(N, dtype=bool)
        C = set()
        while np.any(uncovered):
            best_x = -1
            best_count = -1
            # Optimization: Only consider x that covers at least 1 uncovered point
            for x in range(N):
                cnt = 0
                for nb in balls[x]:
                    if uncovered[nb]:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_x = x
            C.add(best_x)
            for nb in balls[best_x]:
                uncovered[nb] = False
        return C

    def local_search(C, deadline):
        C = set(C)
        cover_count = np.zeros(N, dtype=int)
        for x in C:
            for nb in balls[x]:
                cover_count[nb] += 1
        
        best_C = set(C)
        best_size = len(C)
        
        while time.time() < deadline:
            # Try to remove a random word
            C_list = list(C)
            random.shuffle(C_list)
            removed = False
            for x in C_list:
                can_remove = True
                for nb in balls[x]:
                    if cover_count[nb] == 1:
                        can_remove = False
                        break
                if can_remove:
                    C.remove(x)
                    for nb in balls[x]:
                        cover_count[nb] -= 1
                    removed = True
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = set(C)
                    break
            
            if not removed:
                # Escape local minimum
                # Add a random word
                x_add = random.randint(0, N - 1)
                if x_add not in C:
                    C.add(x_add)
                    for nb in balls[x_add]:
                        cover_count[nb] += 1
                    
                    # Clean up
                    C_list = list(C)
                    random.shuffle(C_list)
                    for x in C_list:
                        if x not in C: continue
                        can_remove = True
                        for nb in balls[x]:
                            if cover_count[nb] == 1:
                                can_remove = False
                                break
                        if can_remove:
                            C.remove(x)
                            for nb in balls[x]:
                                cover_count[nb] -= 1
                            if len(C) < best_size:
                                best_size = len(C)
                                best_C = set(C)
        
        return best_C

    deadline = time.time() + 110
    best_code = greedy_init()
    
    while time.time() < deadline:
        if random.random() < 0.05:
            current = greedy_init()
        else:
            current = set(best_code)
        
        improved = local_search(current, deadline)
        if len(improved) < len(best_code):
            best_code = improved
            
    result = [int_to_word(x) for x in best_code]
    return result

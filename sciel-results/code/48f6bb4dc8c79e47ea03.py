import time
import random
import math

def construct():
        N = 512
        # Precompute conflicts
        # conflicts[x] is a list of masks. Each mask represents 3 other numbers in a 4-AP with x.
        # We use 0-indexed bits for simplicity? No, 1-indexed is fine if we shift.
        # Bit i corresponds to number i.
        
        # Generate all 4-APs
        # a, a+d, a+2d, a+3d
        # a in 1..N, d in 1..(N-a)//3
        # For each AP, update conflicts for each member.
        
        conflicts = [[] for _ in range(N+1)]
        
        for a in range(1, N+1):
            max_d = (N - a) // 3
            for d in range(1, max_d + 1):
                b = a + d
                c = a + 2*d
                e = a + 3*d
                # AP: a, b, c, e
                
                # For a: others are b, c, e
                m = (1 << b) | (1 << c) | (1 << e)
                conflicts[a].append(m)
                
                # For b: others are a, c, e
                m = (1 << a) | (1 << c) | (1 << e)
                conflicts[b].append(m)
                
                # For c: others are a, b, e
                m = (1 << a) | (1 << b) | (1 << e)
                conflicts[c].append(m)
                
                # For e: others are a, b, c
                m = (1 << a) | (1 << b) | (1 << c)
                conflicts[e].append(m)
                
        # Greedy Initialization
        # We can try a few random greedy orders and pick the best
        best_S = set()
        best_size = 0
        
        start_time = time.time()
        deadline = start_time + 55.0 # 5s margin
        
        # Try a few greedy starts
        for attempt in range(20):
            if time.time() > deadline - 45.0: # Leave time for SA
                break
                
            order = list(range(1, N+1))
            random.shuffle(order)
            S = set()
            S_bits = 0
            for x in order:
                # Check if x can be added
                valid = True
                for m in conflicts[x]:
                    if (S_bits & m) == m:
                        valid = False
                        break
                if valid:
                    S.add(x)
                    S_bits |= (1 << x)
            
            if len(S) > best_size:
                best_size = len(S)
                best_S = S
                best_S_bits = S_bits

        # Simulated Annealing on the best greedy solution
        S = best_S
        S_bits = best_S_bits
        size = len(S)
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        last_improvement = time.time()
        
        while time.time() < deadline:
            # Pick a random move
            # 80% chance try to add, 20% chance try to remove (if stuck)
            # Actually, just pick random element
            x = random.randint(1, N)
            
            if x in S:
                # Remove x
                # Delta = -1
                if random.random() < math.exp(-1 / T):
                    S.remove(x)
                    S_bits &= ~(1 << x)
                    size -= 1
            else:
                # Try to add x
                valid = True
                for m in conflicts[x]:
                    if (S_bits & m) == m:
                        valid = False
                        break
                
                if valid:
                    S.add(x)
                    S_bits |= (1 << x)
                    size += 1
                    last_improvement = time.time()
                else:
                    # Optional: try to resolve conflict?
                    # For now, just reject.
                    pass
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
            
            # If stuck for too long, kick
            if time.time() - last_improvement > 2.0:
                # Remove 5 random elements
                for _ in range(5):
                    if S:
                        rem = random.choice(list(S))
                        S.remove(rem)
                        S_bits &= ~(1 << rem)
                        size -= 1
                last_improvement = time.time()
                T = 1.0 # Reheat
        
        return sorted(S)

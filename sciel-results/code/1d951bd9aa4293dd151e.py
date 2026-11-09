import random
import time

def construct():
        best_set = []
        start_time = time.time()
        limit = 55.0 # Leave 5s buffer
        
        while time.time() - start_time < limit:
            pool = list(range(1, 4097))
            random.shuffle(pool)
            current_set = []
            sums_set = set()
            
            for x in pool:
                # Check if x can be added
                new_sums = []
                
                # 3x
                new_sums.append(3*x)
                
                # 2x + s
                for s in current_set:
                    new_sums.append(2*x + s)
                    
                # x + si + sj
                k = len(current_set)
                for i in range(k):
                    si = current_set[i]
                    for j in range(i, k):
                        sj = current_set[j]
                        new_sums.append(x + si + sj)
                        
                # Check internal duplicates
                if len(new_sums) != len(set(new_sums)):
                    continue
                    
                # Check against existing
                valid = True
                for s_val in new_sums:
                    if s_val in sums_set:
                        valid = False
                        break
                        
                if valid:
                    current_set.append(x)
                    sums_set.update(new_sums)
                    
            if len(current_set) > len(best_set):
                best_set = current_set
                
        return best_set

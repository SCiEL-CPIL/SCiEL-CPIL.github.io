import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Greedy Large First
    S_list = []
    S_set = set()
    counts = np.zeros(2001, dtype=int)
    S_arr = np.array([], dtype=int)
    
    for v in range(1000, 0, -1):
        # Check validity
        ok = True
        if counts[2*v] >= 2:
            ok = False
        else:
            if len(S_arr) > 0:
                sums = S_arr + v
                if np.any(counts[sums] >= 2):
                    ok = False
        
        if ok:
            S_list.append(v)
            S_set.add(v)
            if len(S_arr) > 0:
                counts[S_arr + v] += 1
            counts[2*v] += 1
            S_arr = np.append(S_arr, v)
            
    best_S_list = S_list[:]
    best_S_set = S_set.copy()
    best_S_arr = S_arr.copy()
    best_counts = counts.copy()
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        improved = False
        
        # Try to add
        for _ in range(50):
            v = rng.randint(1, 1000)
            if v in best_S_set:
                continue
            
            ok = True
            if best_counts[2*v] >= 2:
                ok = False
            else:
                if len(best_S_arr) > 0:
                    sums = best_S_arr + v
                    if np.any(best_counts[sums] >= 2):
                        ok = False
            
            if ok:
                # Add
                best_S_list.append(v)
                best_S_set.add(v)
                if len(best_S_arr) > 0:
                    best_counts[best_S_arr + v] += 1
                best_counts[2*v] += 1
                best_S_arr = np.append(best_S_arr, v)
                improved = True
        
        # Try to swap
        if len(best_S_list) > 0:
            for _ in range(50):
                idx = rng.randint(0, len(best_S_list)-1)
                x = best_S_list[idx]
                y = rng.randint(1, 1000)
                if x == y or y in best_S_set:
                    continue
                
                # Check if S \ {x} U {y} is valid
                # We need to check if y can be added to S \ {x}
                
                # Temporarily remove x from counts
                # sums involving x: x + u for u in S, u != x. And 2x.
                # We can just recompute or update. Updating is faster.
                
                # Update counts for removal of x
                # S_without_x = best_S_arr without x
                # This is tricky with NumPy arrays.
                # Let's just use a list for S in local search for easier removal, 
                # but use NumPy for the check.
                
                # Actually, let's just do the check on the current array, 
                # but mask out x? No, x is in the array.
                # Sums = best_S_arr + y.
                # We need to ignore the sum x+y when checking?
                # No, the condition is: for all u in S, u != x, counts[u+y] < 2.
                # And counts[2y] < 2.
                
                # Let's construct S_without_x_arr
                mask = best_S_arr != x
                S_without_x_arr = best_S_arr[mask]
                
                ok = True
                if best_counts[2*y] >= 2:
                    # Wait, best_counts[2*y] might be affected by x if x == y? No x!=y.
                    # But if x + x = 2y? i.e. 2x = 2y => x=y. No.
                    # So 2y count is not affected by removing x, unless x=y.
                    # So best_counts[2*y] is correct for S \ {x} regarding 2y.
                    # EXCEPT: if 2x = 2y? No.
                    # So we can use best_counts[2*y].
                    ok = False
                else:
                    if len(S_without_x_arr) > 0:
                        sums = S_without_x_arr + y
                        # We need to check counts[sums].
                        # But best_counts[sums] includes the pair (x, y) if x is in S?
                        # No, y is not in S. So (x,y) is not a pair in S.
                        # So best_counts[sums] is the count for S.
                        # When we remove x, the counts for sums involving x decrease by 1.
                        # sums = S_without_x + y.
                        # Does any sum in `sums` involve x?
                        # No, because x is not in S_without_x.
                        # So best_counts[sums] is the correct count for S \ {x}.
                        
                        if np.any(best_counts[sums] >= 2):
                            ok = False
                
                if ok:
                    # Perform Swap
                    # Remove x
                    if len(best_S_arr) > 0:
                        # Subtract counts for x
                        # Sums: x + u for u in S, u != x. And 2x.
                        # S_without_x_arr is S \ {x}.
                        # counts[x + u] -= 1 for u in S_without_x_arr
                        # counts[2x] -= 1
                        best_counts[best_S_arr + x] -= 1 # This subtracts 2 from x+x if x in S?
                        # Wait, best_S_arr contains x.
                        # best_S_arr + x includes x+x.
                        # The pair (x,x) contributes 1 to counts[2x].
                        # The pair (x,u) for u!=x contributes 1 to counts[x+u].
                        # So best_counts[best_S_arr + x] -= 1 is correct.
                        
                    best_S_arr = best_S_arr[best_S_arr != x]
                    best_S_list.remove(x)
                    best_S_set.discard(x)
                    
                    # Add y
                    if len(best_S_arr) > 0:
                        best_counts[best_S_arr + y] += 1
                    best_counts[2*y] += 1
                    best_S_arr = np.append(best_S_arr, y)
                    best_S_list.append(y)
                    best_S_set.add(y)
                    
                    improved = True
        
        if not improved:
             # Try 1-for-2
             if len(best_S_list) > 0 and rng.random() < 0.5:
                 idx = rng.randint(0, len(best_S_list)-1)
                 x = best_S_list[idx]
                 
                 # Remove x
                 best_counts[best_S_arr + x] -= 1
                 best_S_arr = best_S_arr[best_S_arr != x]
                 best_S_list.remove(x)
                 best_S_set.discard(x)
                 
                 added_any = False
                 # Try to add y
                 for _ in range(10):
                     y = rng.randint(1, 1000)
                     if y in best_S_set or y == x: continue
                     
                     ok = True
                     if best_counts[2*y] >= 2: ok = False
                     else:
                         if len(best_S_arr) > 0:
                             sums = best_S_arr + y
                             if np.any(best_counts[sums] >= 2): ok = False
                     
                     if ok:
                         # Add y
                         if len(best_S_arr) > 0:
                             best_counts[best_S_arr + y] += 1
                         best_counts[2*y] += 1
                         best_S_arr = np.append(best_S_arr, y)
                         best_S_list.append(y)
                         best_S_set.add(y)
                         
                         # Try to add z
                         for _ in range(10):
                             z = rng.randint(1, 1000)
                             if z in best_S_set or z == x or z == y: continue
                             
                             ok2 = True
                             if best_counts[2*z] >= 2: ok2 = False
                             else:
                                 if len(best_S_arr) > 0:
                                     sums2 = best_S_arr + z
                                     if np.any(best_counts[sums2] >= 2): ok2 = False
                             
                             if ok2:
                                 # Add z
                                 if len(best_S_arr) > 0:
                                     best_counts[best_S_arr + z] += 1
                                 best_counts[2*z] += 1
                                 best_S_arr = np.append(best_S_arr, z)
                                 best_S_list.append(z)
                                 best_S_set.add(z)
                                 added_any = True
                                 break
                         
                         if added_any:
                             break
                 
                 if not added_any:
                     # Restore x
                     if len(best_S_arr) > 0:
                         best_counts[best_S_arr + x] += 1
                     best_counts[2*x] += 1
                     best_S_arr = np.append(best_S_arr, x)
                     best_S_list.append(x)
                     best_S_set.add(x)
                     
        if not improved:
             # Check time
             if time.time() - start_time > deadline - 5.0:
                 break

    return best_S_list

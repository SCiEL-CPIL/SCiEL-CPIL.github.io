import sys
import numpy as np

def construct():
    max_val = 1000001
    
    # Initialize largest prime factor array
    # Using a list of ints is standard, but for NumPy we might convert later
    # The sieve itself is best done in pure Python lists for speed in this specific pattern
    # unless we use a more complex vectorized sieve which is hard to write correctly for LPF.
    
    lpf = list(range(max_val + 1))
    
    for i in range(2, max_val + 1):
        if lpf[i] == i:
            for j in range(i, max_val + 1, i):
                lpf[j] = i
                
    # Convert to NumPy array for vectorized operations
    # We only need indices 10000 to 1000000
    # lpf[n] and lpf[n+1]
    # Let's create the array of values for n in range
    # n goes from 10000 to 1000000
    # We need lpf[10000:1000001] and lpf[100001:1000002]
    
    lpf_np = np.array(lpf, dtype=np.int32)
    
    # Extract relevant slices
    # n is in [10000, 1000000]
    # n+1 is in [100001, 1000001]
    # lpf_np indices are 0-based, so lpf_np[k] is lpf for k.
    
    # Slice for n: 10000 to 1000000 inclusive -> 900001 elements
    # Slice for n+1: 100001 to 1000001 inclusive -> 900001 elements
    
    lpf_n = lpf_np[10000:1000001]
    lpf_np1 = lpf_np[100001:1000002]
    
    # Compute max of the two for each pair
    # This creates a new array of size 900001
    max_lpfs = np.maximum(lpf_n, lpf_np1)
    
    # Find the minimum value
    min_val = np.min(max_lpfs)
    
    # Find the index of the first occurrence of this minimum
    # argmin returns the index in the sliced array
    idx = np.argmin(max_lpfs)
    
    # The actual n is 10000 + idx
    best_n = 10000 + idx
    
    return best_n

if __name__ == "__main__":
    print(construct())

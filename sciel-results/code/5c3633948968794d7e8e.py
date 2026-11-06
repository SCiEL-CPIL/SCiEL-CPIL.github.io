   import numpy as np
   
   def is_valid(network):
       # 2^18 inputs
       inputs = np.arange(1 << 18, dtype=np.int64)
       for i, j in network:
           wi = (inputs >> i) & 1
           wj = (inputs >> j) & 1
           min_w = np.minimum(wi, wj)
           max_w = np.maximum(wi, wj)
           inputs = (inputs & ~(1 << i)) | (min_w << i)
           inputs = (inputs & ~(1 << j)) | (max_w << j)
       
       # Check if sorted
       # For each input, check if bits are non-decreasing
       # i.e., input == sorted(input)
       # We can check: for all k, bit k <= bit k+1
       # Or simply: inputs must be of the form 000...0111...1
       # A number is sorted if it has at most one transition from 0 to 1.
       # Or: inputs == (inputs & -inputs) - 1? No.
       
       # Efficient check:
       # A binary number is sorted (non-decreasing) if it looks like 0...01...1
       # This means it is of the form 2^k - 1 (all 1s) or 0 or 2^k - 2^(k-m)?
       # Actually, if we have 18 bits, a sorted number has a specific structure.
       # Let's just check:
       # diff = (inputs >> 1) ^ inputs
       # If sorted, the transition 0->1 happens at most once.
       # So (inputs >> 1) & ~inputs should have at most one bit set?
       # Or simpler:
       # sorted_mask = 0
       # for k in range(18):
       #    ...
       
       # Vectorized check:
       # A number x is sorted if x == 0 or (x & (x+1)) == 0? No.
       # x = 0b001111. x+1 = 0b010000. x & (x+1) = 0.
       # x = 0b0101. x+1 = 0b0110. x & (x+1) = 0b0100 != 0.
       # So condition: (x & (x + 1)) == 0 works for numbers that are "all 1s then all 0s"?
       # Wait, sorted means 0...01...1.
       # Example: 0b00111 (7). 7 & 8 = 0.
       # Example: 0b01111 (15). 15 & 16 = 0.
       # Example: 0b11111 (31). 31 & 32 = 0.
       # Example: 0b00000 (0). 0 & 1 = 0.
       # Example: 0b10000 (16). 16 & 17 = 16 != 0. (16 is 10000, which is NOT sorted in 0..18? 
       # Wait, "sorted" usually means non-decreasing.
       # 10000 is 1,0,0,0,0. Decreasing. Not sorted.
       # 00001 is 0,0,0,0,1. Sorted.
       # So we want numbers like 000...011...1.
       # These numbers satisfy `x & (x + 1) == 0`.
       
       valid = (inputs & (inputs + 1)) == 0
       return np.all(valid)
   
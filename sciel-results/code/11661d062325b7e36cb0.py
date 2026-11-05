  import time
  import numpy as np

  N = 16

  def construct():
      start_time = time.time()
      deadline = start_time + 80  # 80s limit, 10s margin
      
      # Generate Odd-Even Transposition Network
      net = [(i, i + 1) for phase in range(N)
             for i in range(phase % 2, N - 1, 2)]
      
      # Verification function
      def verify(net):
          # Create all 2^N inputs
          # Use np.arange(2**N).reshape(-1, 1)
          # But we need 16 bits.
          inputs = np.arange(2**N, dtype=np.uint16)
          # Expand to 16 columns
          # bits = (inputs[:, None] >> np.arange(N)[::-1]) & 1
          # This is slow in Python loop? No, vectorized.
          
          # Actually, we can just check if the network sorts the inputs.
          # We need to apply the network to each input.
          # Vectorized application:
          # states = inputs.reshape(-1, 1)
          # for i, j in net:
          #   # compare i and j
          #   # min to i, max to j
          #   # This is hard to vectorize easily without masks.
          
          # Alternative: Iterate over inputs? 65536 iterations.
          # Inside, iterate over net (120 comparators).
          # 65536 * 120 = 7.8 million operations.
          # In Python, this is ~1-2 seconds.
          # We might do 10-20 pruning attempts per second.
          # We have 80 seconds. So ~100-200 attempts.
          # Each attempt removes 1 comparator.
          # We need to remove 60.
          # So we can afford ~60 successful removals.
          # This is feasible.
          
          # Let's implement a fast verifier.
          
          # Precompute masks?
          # For each comparator (i, j), we can compute the new state.
          # state_i = state & (1 << i)
          # state_j = state & (1 << j)
          # min_val = state_i & state_j # if both 1, min is 1? No.
          # If we represent 0 as 0 and 1 as 1.
          # min(0,0)=0, min(0,1)=0, min(1,0)=0, min(1,1)=1.
          # max(0,0)=0, max(0,1)=1, max(1,0)=1, max(1,1)=1.
          # min = AND. max = OR.
          # So:
          # new_i = state_i & state_j
          # new_j = state_i | state_j
          
          # We can vectorize this over all 65536 inputs.
          # inputs is an array of shape (65536,) of uint16.
          # We need to extract bits i and j for all inputs.
          # bit_i = (inputs >> i) & 1
          # bit_j = (inputs >> j) & 1
          # new_bit_i = bit_i & bit_j
          # new_bit_j = bit_i | bit_j
          # inputs = (inputs & ~(1<<i) & ~(1<<j)) | (new_bit_i << i) | (new_bit_j << j)
          
          # This is very fast with NumPy.
          
          def check_net(net):
              inputs = np.arange(2**N, dtype=np.uint16)
              for i, j in net:
                  bit_i = (inputs >> i) & 1
                  bit_j = (inputs >> j) & 1
                  new_bit_i = bit_i & bit_j
                  new_bit_j = bit_i | bit_j
                  # Update inputs
                  # Clear bits i and j
                  mask = ~(1 << i) & ~(1 << j)
                  # Note: mask is negative in 2's complement if not careful, but & with uint16 works.
                  # Actually, ~ (1<<i) is a large negative number.
                  # We should use:
                  mask_clear = 0xFFFF ^ (1 << i) ^ (1 << j)
                  inputs = (inputs & mask_clear) | (new_bit_i << i) | (new_bit_j << j)
              
              # Check if sorted
              # A sequence is sorted if for all k, wire[k] <= wire[k+1]
              # In boolean, this means no 1 followed by 0.
              # So we check if there is any pair (k, k+1) such that wire[k]=1 and wire[k+1]=0.
              # bit_k = (inputs >> k) & 1
              # bit_k1 = (inputs >> (k+1)) & 1
              # violation = bit_k & (1 - bit_k1)
              # If any violation is 1 for any input, it's not sorted.
              
              for k in range(N - 1):
                  bit_k = (inputs >> k) & 1
                  bit_k1 = (inputs >> (k + 1)) & 1
                  violations = bit_k & (1 - bit_k1)
                  if np.any(violations > 0):
                      return False
              return True

      # Pruning Loop
      current_net = net
      improved = True
      while improved and time.time() < deadline:
          improved = False
          # Try removing each comparator
          for idx in range(len(current_net)):
              if time.time() > deadline:
                  break
              
              # Try removing comparator at idx
              candidate_net = current_net[:idx] + current_net[idx+1:]
              
              if check_net(candidate_net):
                  current_net = candidate_net
                  improved = True
                  break # Restart loop from beginning with new net
      
      return current_net
  
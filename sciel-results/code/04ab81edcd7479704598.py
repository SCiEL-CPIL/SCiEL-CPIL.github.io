import numpy as np
import time

N = 18

def construct():
          start_time = time.time()
          deadline = 55.0 # seconds

          # Generate seed network
          network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

          # Pre-compute all inputs
          # 2^18 inputs, each 18 bits
          # Use uint8 for efficiency
          num_inputs = 1 << N
          inputs = np.zeros((num_inputs, N), dtype=np.uint8)
          # Fill inputs
          # i in 0..2^18-1
          # bit k of i is input[k]
          # We can use bitwise operations
          # But 2^18 is 262144.
          # We can generate them efficiently.
          # Actually, we can just use a loop or vectorized generation.
          # Let's use a vectorized approach.
          # Create an array of indices 0..2^18-1
          idx = np.arange(num_inputs)
          # Extract bits
          # For each bit position k, (idx >> k) & 1
          for k in range(N):
              inputs[:, k] = (idx >> k) & 1

          # Function to check if network is valid
          def is_valid(net):
              # Simulate network on all inputs
              state = inputs.copy()
              for i, j in net:
                  # min to i, max to j
                  # x_i = min(state[:, i], state[:, j])
                  # x_j = max(state[:, i], state[:, j])
                  # We can do this in place?
                  # No, we need to read both before writing.
                  # But we can use temporary variables or just swap if needed.
                  # Actually, min/max are element-wise.
                  # Let's use np.minimum and np.maximum
                  min_val = np.minimum(state[:, i], state[:, j])
                  max_val = np.maximum(state[:, i], state[:, j])
                  state[:, i] = min_val
                  state[:, j] = max_val

              # Check if sorted
              # state[:, k] <= state[:, k+1]
              # We can check if any state[:, k] > state[:, k+1]
              # Use np.any(np.any(state[:, :-1] > state[:, 1:]))
              # But this creates a large boolean array.
              # Better: check each column pair?
              # Or just:
              for k in range(N - 1):
                  if np.any(state[:, k] > state[:, k + 1]):
                      return False
              return True

          # Pruning loop
          changed = True
          while changed and time.time() < deadline:
              changed = False
              # Try to remove each comparator
              # Iterate in reverse to avoid index issues
              for idx_c in range(len(network) - 1, -1, -1):
                  if time.time() > deadline:
                      break

                  # Remove comparator
                  comparator = network.pop(idx_c)
                  # Check validity
                  if is_valid(network):
                      changed = True
                  else:
                      # Put it back
                      network.insert(idx_c, comparator)

          return network

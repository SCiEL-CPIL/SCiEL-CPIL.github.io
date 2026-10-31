def priority(vector, n):
       # Count non-zeros
       weight = 0
       h = 0
       for v in vector:
           if v != 0:
               weight += 1
           # Simple deterministic hash
           h = (h * 31 + v + 1) % 1000000007
       
       # Prioritize lower weight (Sparse first)
       # Use large multiplier to ensure weight dominates
       # Add hash to break ties randomly
       return float(-weight * 10000000 + h)

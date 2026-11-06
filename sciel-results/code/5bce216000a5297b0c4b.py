      k = p * p
      factor = k
      
      while k <= N:
          pp[k::k] *= factor
          k *= p
          factor = p # Only p for next iterations
  
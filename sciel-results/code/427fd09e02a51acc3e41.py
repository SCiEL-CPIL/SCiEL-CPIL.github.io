  def compute_residuals(params, r):
      U = params[:r*n*m].reshape(n*m, r)
      V = params[r*n*m:r*(n*m+m*p)].reshape(m*p, r)
      W = params[r*(n*m+m*p):].reshape(p*n, r)
      
      residuals = np.zeros(n * p * m * m)
      idx = 0
      for i in range(n):
          for k in range(p):
              w_row = W[k*n + i] # shape (r,)
              for j1 in range(m):
                  u_row = U[i*m + j1] # shape (r,)
                  # Product w_row * u_row is shape (r,)
                  wu = w_row * u_row
                  for j2 in range(m):
                      v_row = V[j2*p + k] # shape (r,)
                      val = np.dot(wu, v_row)
                      target = 1.0 if j1 == j2 else 0.0
                      residuals[idx] = val - target
                      idx += 1
      return residuals
  
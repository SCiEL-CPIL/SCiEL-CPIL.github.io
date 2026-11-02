  def objective(params):
      U_flat = params[:size_u]
      V_flat = params[size_u:size_u + size_v]
      W_flat = params[size_u + size_v:]
      
      U = U_flat.reshape(n * m, rank)
      V = V_flat.reshape(m * p, rank)
      W = W_flat.reshape(p * n, rank)
      
      # Sum over c
      # M is zero tensor of shape (n, m, p)
      M = np.zeros((n, m, p))
      for c in range(rank):
          u = U[:, c].reshape(n, m)
          v = V[:, c].reshape(m, p)
          w = W[:, c].reshape(p, n)
          # Term is u[i,j] * v[j,k] * w[k,i]
          # We can do: temp[i,j,k] = u[i,j] * v[j,k]
          # Then M[i,j,k] += temp[i,j,k] * w[k,i]
          # Using einsum:
          M += np.einsum('ij,jk,ki->ijk', u, v, w)
          
      error = np.sum((M - 1.0) ** 2)
      return error
  
   def khatri_rao(A, B):
       # A: (N, R), B: (M, R)
       # Return: (N*M, R)
       return np.einsum('ir,jr->ijr', A.T, B.T).reshape(-1, R)
   
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
      n, m, p = DIMENSIONS
      rank = 189
      U = [[0] * rank for _ in range(n * m)]
      V = [[0] * rank for _ in range(m * p)]
      W = [[0] * rank for _ in range(p * n)]
      
      # Define Strassen steps
      # Each step: (alpha_blocks, beta_blocks, gamma_blocks)
      # alpha_blocks: list of (r_A, c_A, coeff)
      # beta_blocks: list of (r_B, c_B, coeff)
      # gamma_blocks: list of (r_C, c_C, coeff)
      
      steps = [
          # 1
          ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
          # 2
          ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (1,0,1), (1,1,-1) ]),
          # 3
          ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ], [ (0,1,1), (1,1,1) ]),
          # 4
          ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ], [ (0,0,1), (1,0,1) ]),
          # 5
          ([ (0,0,1), (0,1,1) ], [ (1,1,1) ], [ (0,0,-1), (0,1,1) ]),
          # 6
          ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ], [ (1,1,1) ]),
          # 7
          ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ], [ (0,0,1) ])
      ]
      
      for s, (alphas, betas, gammas) in enumerate(steps):
          for u in range(3):
              for v in range(3):
                  for w in range(3):
                      r = s * 27 + u * 9 + v * 3 + w
                      
                      # U
                      for r_A, c_A, coeff in alphas:
                          i = r_A * 3 + u
                          j = c_A * 3 + v
                          U[i * m + j][r] += coeff
                          
                      # V
                      for r_B, c_B, coeff in betas:
                          j = r_B * 3 + v
                          k = c_B * 3 + w
                          V[j * p + k][r] += coeff
                          
                      # W
                      for r_C, c_C, coeff in gammas:
                          i = r_C * 3 + u
                          k = c_C * 3 + w
                          W[k * n + i][r] += coeff
                          
      return {"U": U, "V": V, "W": W}

      def validate(positions, radii, w, h, tol=1e-9):
          for i in range(N):
              if positions[i, 0] - radii[i] < -tol or positions[i, 0] + radii[i] > w + tol:
                  return False
              if positions[i, 1] - radii[i] < -tol or positions[i, 1] + radii[i] > h + tol:
                  return False
              for j in range(i + 1, N):
                  dx = positions[j, 0] - positions[i, 0]
                  dy = positions[j, 1] - positions[i, 1]
                  dist = np.sqrt(dx*dx + dy*dy)
                  if dist < radii[i] + radii[j] - tol:
                      return False
          return True
      
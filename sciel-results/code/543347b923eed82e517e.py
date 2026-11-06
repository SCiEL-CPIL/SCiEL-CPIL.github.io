def construct():
      S = []
      for x in range(256, 0, -1):
          valid = True
          # Check if x divides b+c for any b, c in S
          # b, c must be >= x? No, b, c are in S, which are > x.
          # So b > x and c > x.
          # Condition: x < b <= c.
          # Since b, c > x, x < b is satisfied.
          # We just need to check if x divides b+c.
          n = len(S)
          for i in range(n):
              b = S[i]
              for j in range(i, n):
                  c = S[j]
                  # b <= c is ensured by i <= j if S is sorted?
                  # S is built by adding smaller numbers, so S is not sorted ascending.
                  # S contains numbers > x.
                  # We just need to check all pairs.
                  if (b + c) % x == 0:
                      valid = False
                      break
              if not valid:
                  break
          if valid:
              S.append(x)
      return S

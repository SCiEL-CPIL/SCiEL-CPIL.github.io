  for i in range(3):
      for j in range(3):
          for k in range(3):
              if n_parts[i] == 2 and m_parts[j] == 2 and p_parts[k] == 2:
                  total_rank += 7
              else:
                  total_rank += n_parts[i] * m_parts[j] * p_parts[k]
  
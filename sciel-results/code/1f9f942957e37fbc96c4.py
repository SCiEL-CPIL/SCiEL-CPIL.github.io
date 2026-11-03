      improved = True
      while improved and time.time() < deadline:
          improved = False
          for i in range(len(current)):
              if time.time() > deadline: break
              saved = current.pop(i)
              if is_valid(current):
                  improved = True
              else:
                  current.insert(i, saved)
      